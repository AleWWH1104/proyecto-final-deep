"""Rutas de la interfaz web: solo devuelven los archivos HTML de app/web."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

DIR_WEB = Path(__file__).resolve().parents[1] / "web"

router = APIRouter(include_in_schema=False)


@router.get("/")
def inicio() -> FileResponse:
    return FileResponse(DIR_WEB / "index.html")


@router.get("/monitoreo.html")
def panel_monitoreo() -> FileResponse:
    return FileResponse(DIR_WEB / "monitoreo.html")
