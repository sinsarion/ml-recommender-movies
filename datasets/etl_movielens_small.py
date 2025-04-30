import pandas as pd
from sqlalchemy import create_engine, text

POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'secret'
POSTGRES_DB = 'ml-db'
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432

SCHEMA = "movielens_small"

DATA_DIR = f"./{SCHEMA}"  # путь к распакованным CSV-файлам датасета
CHUNK_SIZE = 50000

engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")


def run_query(query):
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()


def disable_constraints():
    print("Отключаем проверку foreign key...")
    run_query(f"SET session_replication_role = replica;")


def enable_constraints():
    print("Включаем проверку foreign key...")
    run_query(f"SET session_replication_role = DEFAULT;")


def load_users(ratings_df):
    users_df = pd.DataFrame({'id': ratings_df['user_id'].unique()})
    users_df.to_sql('users', engine, schema=SCHEMA, if_exists='append', index=False, method='multi')


def load_movies(movies_df):
    movies_df.to_sql('movies', engine, schema=SCHEMA, if_exists='append', index=False, method='multi')


def load_genres(movies_df):
    genre_set = set()
    for genres in movies_df['raw_genres'].dropna():
        for genre in genres.split('|'):
            genre_set.add(genre)
    genres_df = pd.DataFrame({'name': sorted(genre_set)})
    genres_df.to_sql('genres', engine, schema=SCHEMA, if_exists='append', index=False, method='multi')


def load_movie_genres(movies_df):
    conn = engine.connect()
    genres_df = pd.read_sql_table('genres', conn, schema=SCHEMA)
    movie_genres = []
    for row in movies_df.itertuples():
        movie_id = row.id
        if pd.isna(row.raw_genres):
            continue
        for genre in row.raw_genres.split('|'):
            genre_id = genres_df[genres_df['name'] == genre]['id'].values[0]
            movie_genres.append({'movie_id': movie_id, 'genre_id': genre_id})
    pd.DataFrame(movie_genres).drop_duplicates().to_sql('movie_genres', engine, schema=SCHEMA, if_exists='append',
                                                        index=False, method='multi')


def load_ratings():
    df_iter = pd.read_csv(f"{DATA_DIR}/ratings.csv", chunksize=CHUNK_SIZE)
    for chunk in df_iter:
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='s')
        chunk = chunk.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
        chunk.to_sql('ratings', engine, schema=SCHEMA, if_exists='append', index=False, method='multi')
    all_df = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    all_df = all_df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], unit='s')
    return all_df


def load_tags():
    df = pd.read_csv(f"{DATA_DIR}/tags.csv")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df = df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})

    df = df.dropna(subset=["user_id", "movie_id", "tag", "timestamp"])
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df = df.drop_duplicates(subset=["user_id", "movie_id", "tag", "timestamp"])

    with engine.connect() as conn:
        users = pd.read_sql(f"SELECT id FROM {SCHEMA}.users", conn)['id'].astype(int).unique()
        movies = pd.read_sql(f"SELECT id FROM {SCHEMA}.movies", conn)['id'].astype(int).unique()

    df = df[df['user_id'].isin(users) & df['movie_id'].isin(movies)]

    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {SCHEMA}.tags RESTART IDENTITY CASCADE"))

    df.to_sql('tags', engine, schema=SCHEMA, if_exists='append', index=False, method='multi')


def load_links(chunk_size: int = CHUNK_SIZE):
    df = pd.read_csv(f"{DATA_DIR}/links.csv")

    print("Переименовываем поля movieId в movie_id, imdbId в imdb_id, tmdbId в tmdb_id...")
    df = df.rename(columns={
        'movieId': 'movie_id',
        'imdbId': 'imdb_id',
        'tmdbId': 'tmdb_id'
    })

    print("Удаляем пустые значения...")
    df = df.dropna(subset=["movie_id"])
    df["movie_id"] = df["movie_id"].astype(int)

    print("Удаляем дубликаты...")
    df = df.drop_duplicates(subset=["movie_id"])

    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").dropna().astype("Int64")

    with engine.connect() as conn:
        print("Читаем список существующих movie_id из БД...")
        movies = pd.read_sql(f"SELECT id FROM {SCHEMA}.movies", conn)['id'].astype(int).unique()

    df = df[df['movie_id'].isin(movies)]

    print("TRUNCATE таблицы links...")
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {SCHEMA}.links RESTART IDENTITY CASCADE"))

    print(f"Вставка по чанкам: {chunk_size} записей за раз")
    inserted = 0

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        try:
            chunk.to_sql('links', engine, schema=SCHEMA, if_exists='append', index=False, method='multi')
            inserted += len(chunk)
            print(f"Вставлено {inserted} / {len(df)}")
        except Exception as e:
            print(f"Ошибка при вставке чанка с {i} по {i + chunk_size}: {str(e)}")
            break

    print("Загрузка links завершена.")


def main():
    disable_constraints()

    print("Загружаем ratings...")
    ratings_df = load_ratings()

    print("Загружаем users...")
    load_users(ratings_df)

    print("Загружаем movies...")
    movies_df = pd.read_csv(f"{DATA_DIR}/movies.csv")
    movies_df = movies_df.rename(columns={'movieId': 'id', 'genres': 'raw_genres'})
    load_movies(movies_df)

    print("Загружаем genres...")
    load_genres(movies_df)

    print("Загружаем movie_genres...")
    load_movie_genres(movies_df)

    print("Загружаем tags...")
    load_tags()

    print("Загружаем links...")
    load_links()

    enable_constraints()

    print(f"Данные успешно загружены в схему: {SCHEMA}.")


if __name__ == "__main__":
    main()
