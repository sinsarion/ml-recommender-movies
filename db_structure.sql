CREATE SCHEMA IF NOT EXISTS movielens_32m;
CREATE SCHEMA IF NOT EXISTS movielens_small;

-- Таблица пользователей movielens_small
CREATE TABLE IF NOT EXISTS movielens_small.users
(
    id INTEGER PRIMARY KEY
);

-- Таблица фильмов movielens_small
CREATE TABLE IF NOT EXISTS movielens_small.movies
(
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    -- Жанры будут вынесены в отдельную таблицу, но сохраним строку как оригинал
    raw_genres TEXT
);

-- Таблица жанров movielens_small
CREATE TABLE IF NOT EXISTS movielens_small.genres
(
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- Связующая таблица фильмов и жанров (many-to-many) movielens_small
CREATE TABLE IF NOT EXISTS movielens_small.movie_genres
(
    movie_id INTEGER REFERENCES movielens_small.movies (id) ON DELETE CASCADE,
    genre_id INTEGER REFERENCES movielens_small.genres (id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, genre_id)
);

-- Таблица рейтингов movielens_small
CREATE TABLE IF NOT EXISTS movielens_small.ratings
(
    user_id   INTEGER REFERENCES movielens_small.users (id) ON DELETE CASCADE,
    movie_id  INTEGER REFERENCES movielens_small.movies (id) ON DELETE CASCADE,
    rating    FLOAT,
    timestamp TIMESTAMP,
    PRIMARY KEY (user_id, movie_id)
);

-- Таблица тегов movielens_small
CREATE TABLE IF NOT EXISTS movielens_small.tags
(
    user_id   INTEGER REFERENCES movielens_small.users (id) ON DELETE CASCADE,
    movie_id  INTEGER REFERENCES movielens_small.movies (id) ON DELETE CASCADE,
    tag       TEXT,
    timestamp TIMESTAMP,
    PRIMARY KEY (user_id, movie_id, tag)
);

-- Таблица внешних ссылок movielens_small
CREATE TABLE IF NOT EXISTS movielens_small.links
(
    movie_id INTEGER PRIMARY KEY REFERENCES movielens_small.movies (id) ON DELETE CASCADE,
    imdb_id  TEXT,
    tmdb_id  TEXT
);

-- Таблица пользователей movielens_32m
CREATE TABLE IF NOT EXISTS movielens_32m.users
(
    id INTEGER PRIMARY KEY
);

-- Таблица фильмов movielens_32m
CREATE TABLE IF NOT EXISTS movielens_32m.movies
(
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    raw_genres TEXT
);

-- Таблица жанров movielens_32m
CREATE TABLE IF NOT EXISTS movielens_32m.genres
(
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- Таблица связей фильмы - жанры movielens_32m
CREATE TABLE IF NOT EXISTS movielens_32m.movie_genres
(
    movie_id INTEGER REFERENCES movielens_32m.movies (id) ON DELETE CASCADE,
    genre_id INTEGER REFERENCES movielens_32m.genres (id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, genre_id)
);

-- Таблица рейтингов movielens_32m
CREATE TABLE IF NOT EXISTS movielens_32m.ratings
(
    user_id   INTEGER REFERENCES movielens_32m.users (id) ON DELETE CASCADE,
    movie_id  INTEGER REFERENCES movielens_32m.movies (id) ON DELETE CASCADE,
    rating    FLOAT     NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, movie_id, timestamp)
);

-- Таблица тегов movielens_32m
CREATE TABLE IF NOT EXISTS movielens_32m.tags
(
    user_id   INTEGER REFERENCES movielens_32m.users (id) ON DELETE CASCADE,
    movie_id  INTEGER REFERENCES movielens_32m.movies (id) ON DELETE CASCADE,
    tag       TEXT      NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, movie_id, tag, timestamp)
);

-- Таблица ссылок movielens_32m
CREATE TABLE IF NOT EXISTS movielens_32m.links
(
    movie_id INTEGER PRIMARY KEY REFERENCES movielens_32m.movies (id) ON DELETE CASCADE,
    imdb_id  TEXT,
    tmdb_id  TEXT
);

-- Индексы для оптимизации movielens_32m
CREATE INDEX IF NOT EXISTS idx_ratings_user ON movielens_32m.ratings (user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON movielens_32m.ratings (movie_id);
CREATE INDEX IF NOT EXISTS idx_ratings_user_movie ON movielens_32m.ratings (user_id, movie_id);

CREATE INDEX IF NOT EXISTS idx_tags_movie ON movielens_32m.tags (movie_id);
CREATE INDEX IF NOT EXISTS idx_tags_user ON movielens_32m.tags (user_id);

