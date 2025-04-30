import logging

from fastapi import APIRouter, Depends
from fastapi_limiter.depends import RateLimiter
from starlette import status
from starlette.responses import JSONResponse

from services.ml_service import (
    MLRecommenderService,
    get_ml_recommender_service
)

logger = logging.getLogger('uvicorn.error')
router = APIRouter()


@router.get(
    path="/recommendations/{user_id}",
    tags=["ml_api"],
    summary="Персонализированный запрос на получение списка рекомендованных фильмов",
    dependencies=[Depends(RateLimiter(times=2, seconds=1))]  # Лимит: 2 запроса в 1 секунду
)
async def get_recommendations(
        user_id: str,
        ml_model: str,
        dataset: str,
        k: int = 10,
        ml_recommender_service: MLRecommenderService = Depends(get_ml_recommender_service)
):
    """Персонализированный запрос на получение списка рекомендованных фильмов"""
    logger.info(f"get_recommendations() started, user_id = {user_id} dataset = {dataset} k = {k}")

    result = await ml_recommender_service.get_recommendations(user_id, ml_model, dataset, k)

    return JSONResponse(content=result.model_dump(), status_code=status.HTTP_200_OK)


@router.get(
    path="/recommendations",
    tags=["ml_api"],
    summary="Не персонализированный запрос на получение списка рекомендованных фильмов",
    dependencies=[Depends(RateLimiter(times=2, seconds=1))]  # Лимит: 2 запроса в 1 секунду
)
async def get_non_personal_recommendations(
        dataset: str,
        k: int = 10,
        ml_recommender_service: MLRecommenderService = Depends(get_ml_recommender_service)
):
    """Не персонализированный запрос на получение списка рекомендованных фильмов"""
    logger.info(f"get_non_personal_recommendations() started, dataset = {dataset} k = {k}")

    result = await ml_recommender_service.get_non_personalized_recommendations(dataset=dataset, k=k)

    return JSONResponse(content=result.model_dump(), status_code=status.HTTP_200_OK)
