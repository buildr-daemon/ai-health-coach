from fastapi import APIRouter
from api.v1.user import router as user_router
from api.v1.chat import router as chat_router

router = APIRouter(prefix="/v1")

router.include_router(user_router)
router.include_router(chat_router)
