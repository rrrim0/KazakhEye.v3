import json
import logging.config
import os
import shutil
import traceback
from datetime import datetime

import aiohttp_cors
from aiohttp import web
from aiohttp_swagger import setup_swagger
from tensorflow.keras.models import load_model

from services import PredictService

with open("config.json") as file:
    config = json.load(file)

if not os.path.exists("temp"):
    os.makedirs("temp")

logging.config.dictConfig(config["logger"])
logger = logging.getLogger(config["source"])

model = load_model("license_plate_model.h5")

service = PredictService(
    logger=logger,
    config=config,
    model=model
)


async def predict(request):
    directory = None
    try:
        reader = await request.multipart()
        media = await reader.next()

        if not media or media.name != 'photo':
            return web.json_response(status=500)

        directory = os.path.join("temp", datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.exists(directory):
            os.makedirs(directory)

        destination = os.path.join(directory, media.filename)
        with open(destination, "wb") as f:
            while chunk := await media.read_chunk():
                f.write(chunk)

        result = await service.predict(path=destination)
        if "error" in result:
            return web.json_response(status=result["status"], data={"error": result["error"]})
        return web.json_response(status=200, data=result)
    except Exception as e:
        logger.error(f"{e} {traceback.format_exc()}")
        return web.json_response(status=500)
    finally:
        shutil.rmtree(directory) if directory is not None else None


app = web.Application()

app.add_routes([
    web.post('/api/predict', predict),
])

setup_swagger(
    app=app,
    title="License Plate Determinator API",
    description="",
    ui_version=3,
    swagger_url="swagger",
)

cors = aiohttp_cors.setup(
    app=app,
    defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    }
)

for route in app.router.routes():
    cors.add(route)

if __name__ == '__main__':
    logger.info(f"Running app on {config['host']}:{config['port']}")
    web.run_app(app, host=config['host'], port=config['port'])
