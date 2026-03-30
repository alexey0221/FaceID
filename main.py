import cv2
import numpy as np


def expand_bbox(x, y, w, h, image_shape, margin_percent=0.3):
    """
    Расширяет bounding box с учетом полей

    Args:
        x, y, w, h: исходные координаты лица
        image_shape: форма изображения (height, width)
        margin_percent: процент расширения (0.3 = 30% от ширины/высоты)

    Returns:
        new_x, new_y, new_w, new_h: расширенные координаты
    """
    # Вычисляем отступы
    margin_w = int(w * margin_percent)
    margin_h = int(h * margin_percent)

    # Расширяем координаты
    new_x = max(0, x - margin_w)
    new_y = max(0, y - margin_h)
    new_w = min(image_shape[1] - new_x, w + 2 * margin_w)
    new_h = min(image_shape[0] - new_y, h + 2 * margin_h)

    return new_x, new_y, new_w, new_h


def process_face(image, face_cascade, facemark, margin_percent=0.3):
    """
    Обрабатывает изображение: находит лицо, расширяет область, обрезает и расставляет точки

    Args:
        image: исходное изображение
        face_cascade: каскад Хаара
        facemark: модель LBF
        margin_percent: процент расширения области лица
    """
    # Копируем изображение
    img_copy = image.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Поиск лица
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("Лицо не обнаружено!")
        return None, None

    results = []
    h_img, w_img = image.shape[:2]

    for i, (x, y, w, h) in enumerate(faces):
        print(f"Лицо {i + 1}: исходный bounding box: x={x}, y={y}, w={w}, h={h}")

        # Расширяем bounding box
        new_x, new_y, new_w, new_h = expand_bbox(x, y, w, h, (h_img, w_img), margin_percent)
        print(f"Лицо {i + 1}: расширенный bounding box: x={new_x}, y={new_y}, w={new_w}, h={new_h}")

        # Обрезаем расширенную область лица
        face_roi = img_copy[new_y:new_y + new_h, new_x:new_x + new_w]

        # Создаем массив координат для модели (с использованием расширенных координат)
        # Важно: модель LBF ожидает координаты в оригинальном изображении
        faces_array = np.array([[x, y, w, h]])  # Используем оригинальные координаты для модели

        # Поиск опорных точек
        _, landmarks = facemark.fit(img_copy, faces_array)

        # Сохраняем результаты
        results.append({
            'bbox': (x, y, w, h),  # Оригинальный bbox
            'expanded_bbox': (new_x, new_y, new_w, new_h),  # Расширенный bbox
            'face_roi': face_roi,
            'landmarks': landmarks
        })

        # Рисуем прямоугольники на оригинальном изображении
        # Оригинальный bbox - синий
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Расширенный bbox - зеленый
        cv2.rectangle(img_copy, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

        # Рисуем точки на оригинальном изображении
        if landmarks:
            for landmark in landmarks:
                for (point_x, point_y) in landmark[0]:
                    cv2.circle(img_copy, (int(point_x), int(point_y)), 2, (0, 0, 255), -1)

    return img_copy, results


def main():
    # Загрузка моделей
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    facemark = cv2.face.createFacemarkLBF()

    # if not (
    facemark.loadModel("lbfmodel.yaml")#):
        # print("Ошибка: Не удалось загрузить модель LBF")
        # return

    # Загрузка изображения
    image = cv2.imread('face.jpg')
    if image is None:
        print("Ошибка: Не удалось загрузить изображение")
        return

    # Параметры расширения (можно экспериментировать)
    MARGIN_PERCENT = 0.3  # 40% расширения (хорошо для большинства случаев)

    # Обработка изображения
    result_image, processed_faces = process_face(image, face_cascade, facemark, MARGIN_PERCENT)

    if result_image is not None:
        # Отображение результатов
        # cv2.imshow('Original with landmarks (Blue: original, Green: expanded)', result_image)

        # Отображение каждого обрезанного лица
        for i, face_data in enumerate(processed_faces):
            # Создаем копию обрезанного лица для отрисовки точек
            face_with_points = face_data['face_roi'].copy()

            # Рисуем точки на обрезанном лице
            if face_data['landmarks']:
                # Корректируем координаты для обрезанного изображения
                x, y, w, h = face_data['bbox']  # Оригинальный bbox
                new_x, new_y, new_w, new_h = face_data['expanded_bbox']  # Расширенный bbox

                for landmark in face_data['landmarks']:
                    for (point_x, point_y) in landmark[0]:
                        # Преобразуем координаты относительно расширенного обрезанного изображения
                        adj_x = int(point_x - new_x)
                        adj_y = int(point_y - new_y)
                        if 0 <= adj_x < new_w and 0 <= adj_y < new_h:
                            cv2.circle(face_with_points, (adj_x, adj_y), 2, (0, 255, 0), -1)

            # Сохраняем обрезанное лицо
            cv2.imwrite(f'cropped_face_{i + 1}_expanded.jpg', face_with_points)
            cv2.imshow(f'Cropped Face {i + 1} (expanded)', face_with_points)

        # Сохранение результата
        cv2.imwrite('result_with_landmarks.jpg', result_image)
        print("Результаты сохранены!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Не удалось обработать изображение")


if __name__ == "__main__":
    main()