import os
import string
from logging import Logger

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


class PredictService:
    def __init__(self, logger: Logger, config: dict, model):
        self.__logger = logger
        self.__config = config
        self.__model = model
        self.__regions = {
            "01": "г. Нур-Султан (Астана)",
            "02": "г. Алматы",
            "03": "Акмолинская область",
            "04": "Актюбинская область",
            "05": "Алматинская область",
            "06": "Атырауская область",
            "07": "Западно-Казахстанская область",
            "08": "Жамбылская область",
            "09": "Карагандинская область",
            "10": "Костанайская область",
            "11": "Кызылординская область",
            "12": "Мангистауская область",
            "13": "Туркестанская область",
            "14": "Павлодарская область",
            "15": "Северо-Казахстанская область",
            "16": "Восточно-Казахстанская область",
            "17": "г. Шымкент",
            "18": "Абайская область",
            "19": "Жетысуская область",
            "20": "Улытауская область",
        }
        self.__idx_to_char = {idx: char for idx, char in enumerate(string.ascii_uppercase + string.digits)}

    async def predict(self, path: str) -> dict:
        png = self.__to_png(path=path)
        if not png:
            return {"error": "При конвертации файла в .png произошла ошибка", "status": 400}

        image = self.__adapt_image(path=png)
        if image is None:
            return {"error": "При подготовке файла к анализу произошла ошибка", "status": 400}

        prediction = self.__model.predict(image)
        plate = ''.join([self.__idx_to_char[np.argmax(pred)] for pred in prediction[0]]).lower()

        # if "kz" not in plate:
        #     return {"plate": plate}

        region = self.__regions.get(plate[-2:])
        return {"plate": plate, "region": region}

    @staticmethod
    def __adapt_image(path: str):
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 64))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception:
            return None

    @staticmethod
    def __to_png(path: str):
        if path.lower().endswith('.png'):
            return path

        try:
            with Image.open(path) as img:
                base_name = os.path.splitext(path)[0]
                save_path = f"{base_name}.png"
                img = img.convert("RGBA")
                img.save(save_path, format="PNG")
                return save_path
        except Exception:
            return None