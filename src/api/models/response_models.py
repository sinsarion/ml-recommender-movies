from pydantic import BaseModel


class FilmInfo(BaseModel):
    id: int
    title: str
    avg_rating: float | None = None
    genres: list[str] = []
    imdb_id: str | None = None
    tmdb_id: str | None = None
    imdb_url: str | None = None
    tmdb_url: str | None = None


class RecommendationsResponse(BaseModel):
    user_id: str
    has_personalization: bool
    recommendations: list[FilmInfo]
