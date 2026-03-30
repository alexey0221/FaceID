import cv2
import numpy as np
import math
import json
import os
import sys


def calculate_distance(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def expand_bbox(x, y, w, h, image_shape, margin_percent=0.3):
    """Расширяет bounding box с учетом полей"""
    margin_w = int(w * margin_percent)
    margin_h = int(h * margin_percent)

    new_x = max(0, x - margin_w)
    new_y = max(0, y - margin_h)
    new_w = min(image_shape[1] - new_x, w + 2 * margin_w)
    new_h = min(image_shape[0] - new_y, h + 2 * margin_h)

    return new_x, new_y, new_w, new_h


def process_face(image, face_cascade, facemark, margin_percent=0.3):
    """Обрабатывает изображение: находит лицо, расширяет область, обрезает и расставляет точки"""
    img_copy = image.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("Лицо не обнаружено!")
        return None, None

    results = []
    h_img, w_img = image.shape[:2]

    for i, (x, y, w, h) in enumerate(faces):
        new_x, new_y, new_w, new_h = expand_bbox(x, y, w, h, (h_img, w_img), margin_percent)
        face_roi = img_copy[new_y:new_y + new_h, new_x:new_x + new_w]

        faces_array = np.array([[x, y, w, h]])
        _, landmarks = facemark.fit(img_copy, faces_array)

        results.append({
            'bbox': (x, y, w, h),
            'expanded_bbox': (new_x, new_y, new_w, new_h),
            'face_roi': face_roi,
            'landmarks': landmarks
        })

        if landmarks:
            for landmark in landmarks:
                for (point_x, point_y) in landmark[0]:
                    cv2.circle(img_copy, (int(point_x), int(point_y)), 2, (0, 0, 255), -1)

    return img_copy, results


def build_custom_face_graph(landmarks):
    """
    Строит граф на основе пользовательских правил
    """
    if not landmarks or len(landmarks) < 68:
        return None, None

    points = np.array(landmarks)
    connections = []

    # ==================== 1. КОНТУР ЛИЦА ====================
    for i in range(16):
        connections.append((i, i + 1))

    # ==================== 2. БРОВИ ====================
    for i in range(17, 21):
        connections.append((i, i + 1))
    for i in range(22, 26):
        connections.append((i, i + 1))

    # ==================== 3. НОС (перевернутая буква Т) ====================
    connections.extend([
        (27, 28), (28, 29), (29, 30), (30, 32),
        (31, 32), (32, 33)
    ])

    # ==================== 4. ГЛАЗА ====================
    connections.extend([
        (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
        (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)
    ])

    # ==================== 5. ГУБЫ ====================
    for i in range(48, 59):
        connections.append((i, i + 1))
    connections.append((59, 48))

    for i in range(60, 67):
        connections.append((i, i + 1))
    connections.append((67, 60))

    # ==================== 6. СВЯЗИ: ВНУТРЕННИЕ УГЛЫ ГЛАЗ К ПЕРЕНОСИЦЕ ====================
    connections.extend([
        (39, 27), (40, 27), (42, 27), (47, 27)
    ])

    # ==================== 7. СВЯЗИ: ВНЕШНИЕ УГЛЫ ГЛАЗ К КОНТУРУ ЛИЦА ====================
    connections.extend([
        (36, 0), (36, 1), (36, 2), (36, 3),
        (38, 3), (39, 4),
        (45, 16), (45, 15),
        (46, 15), (46, 14), (46, 13)
    ])

    # ==================== 8. СВЯЗИ: БРОВИ К ВЕРХНИМ ТОЧКАМ ГЛАЗ ====================
    connections.extend([
        (18, 37), (19, 38), (20, 39),
        (23, 44), (24, 43), (25, 45)
    ])

    # ==================== 9. СВЯЗИ: НОЗДРИ К КОНТУРУ ЛИЦА ====================
    connections.extend([
        (31, 6), (33, 10)
    ])

    # ==================== 10. СВЯЗИ: УГОЛКИ ГУБ К КОНТУРУ ЛИЦА ====================
    connections.extend([
        (48, 4), (48, 5), (50, 7),
        (54, 11), (54, 12), (54, 13),
        (57, 10), (57, 8), (57, 9)
    ])

    # ==================== ДОПОЛНИТЕЛЬНЫЕ СВЯЗИ ====================
    connections.extend([
        (17, 0), (22, 13),
        (27, 21), (27, 22),
        (30, 39), (30, 42)
    ])

    # Удаляем дубликаты
    connections = list(set(connections))

    # Создаем граф с весами
    n_points = len(points)
    edges = []

    for i, j in connections:
        if i < n_points and j < n_points:
            dist = calculate_distance(points[i], points[j])
            edges.append((i, j, dist))

    return edges


def extract_edge_features(landmarks, edges, reference_edge=(39, 4)):
    """
    Извлекает характеристики ребер: абсолютные длины и относительные
    """
    # Находим длину эталонного ребра
    ref_length = None
    for i, j, dist in edges:
        if (i == reference_edge[0] and j == reference_edge[1]) or \
                (i == reference_edge[1] and j == reference_edge[0]):
            ref_length = dist
            break

    if ref_length is None:
        ref_length = 1.0

    # Создаем словарь с данными
    edge_features = {}

    for idx, (i, j, abs_length) in enumerate(edges):
        edge_key = f"{min(i, j)}_{max(i, j)}"
        relative_length = abs_length / ref_length

        edge_features[edge_key] = {
            'relative_length': relative_length
        }

    return edge_features, ref_length


def authenticate_face(image_path, user_data_file, tolerance=0.05):
    """
    Проверяет аутентификацию по фотографии

    Args:
        image_path: путь к фотографии для проверки
        user_data_file: путь к файлу с данными пользователя
        tolerance: допустимое отклонение относительных длин

    Returns:
        dict: результаты аутентификации
    """
    # Загружаем данные пользователя
    if not os.path.exists(user_data_file):
        return {
            'success': False,
            'authenticated': False,
            'error': f"Файл {user_data_file} не найден"
        }

    with open(user_data_file, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    # Загружаем модели
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    facemark = cv2.face.createFacemarkLBF()

    facemark.loadModel("lbfmodel.yaml")#:
    # return {
    #     'success': False,
    #     'authenticated': False,
    #     'error': "Не удалось загрузить модель LBF"
    # }

# Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        return {
            'success': False,
            'authenticated': False,
            'error': f"Не удалось загрузить изображение {image_path}"
        }

    # Обрабатываем изображение
    MARGIN_PERCENT = 0.3
    result_image, processed_faces = process_face(image, face_cascade, facemark, MARGIN_PERCENT)

    if result_image is None or not processed_faces:
        return {
            'success': False,
            'authenticated': False,
            'error': "Не удалось обнаружить лицо на изображении"
        }

    # Получаем точки лица
    landmarks = []
    face_data = processed_faces[0]

    if face_data['landmarks']:
        for landmark in face_data['landmarks']:
            for (x, y) in landmark[0]:
                landmarks.append((float(x), float(y)))

    if len(landmarks) < 68:
        return {
            'success': False,
            'authenticated': False,
            'error': f"Обнаружено недостаточно точек: {len(landmarks)}"
        }

    # Строим граф
    edges = build_custom_face_graph(landmarks)

    if not edges:
        return {
            'success': False,
            'authenticated': False,
            'error': "Не удалось построить граф лица"
        }

    # Извлекаем характеристики ребер
    reference_edge = (39, 4)
    current_features, ref_length = extract_edge_features(landmarks, edges, reference_edge)

    # Сравниваем с эталоном
    stored_edges = user_data['edges']
    common_edges = set(current_features.keys()) & set(stored_edges.keys())

    if len(common_edges) == 0:
        return {
            'success': True,
            'authenticated': False,
            'error': "Нет общих ребер для сравнения",
            'similarity': 0,
            'max_error': 1.0
        }

    # Вычисляем ошибки для каждого ребра
    errors = []
    edge_errors = {}

    for edge_key in common_edges:
        stored_rel_len = stored_edges[edge_key]['relative_length']
        current_rel_len = current_features[edge_key]['relative_length']
        error = abs(stored_rel_len - current_rel_len)
        errors.append(error)
        edge_errors[edge_key] = error

    max_error = max(errors)
    avg_error = np.mean(errors)
    similarity = 1 - min(avg_error / tolerance, 1)
    authenticated = avg_error < tolerance

    # Визуализируем результат на изображении
    img_result = result_image.copy()

    # Рисуем точки
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_result, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Рисуем ребра с цветовой индикацией ошибок
    for edge_key in common_edges:
        edge_data = stored_edges[edge_key]
        point1_idx = edge_data['point1']
        point2_idx = edge_data['point2']
        error = edge_errors[edge_key]

        point1 = (int(landmarks[point1_idx][0]), int(landmarks[point1_idx][1]))
        point2 = (int(landmarks[point2_idx][0]), int(landmarks[point2_idx][1]))

        # Цвет ребра в зависимости от ошибки
        if error < tolerance * 0.3:
            color = (0, 255, 0)  # Зеленый - малая ошибка
        elif error < tolerance * 0.7:
            color = (0, 255, 255)  # Желтый - средняя ошибка
        else:
            color = (0, 0, 255)  # Красный - большая ошибка

        cv2.line(img_result, point1, point2, color, 2)

    # Результат аутентификации
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ АУТЕНТИФИКАЦИИ")
    print("=" * 60)

    if authenticated:
        print(f"\n✅ Аутентификация: УСПЕШНО")
    else:
        print(f"\n❌ Аутентификация: НЕУСПЕШНО")

    print(f"\n📊 Детальная статистика:")
    print(f"  • Максимальная ошибка: {max_error:.4f}")
    print(f"  • Средняя ошибка: {avg_error:.4f}")
    print(f"  • Порог: {tolerance:.4f}")
    print(f"  • Схожесть: {similarity * 100:.1f}%")

    print("\n" + "=" * 60)
    if authenticated:
        print("✅ ДОСТУП РАЗРЕШЕН")
    else:
        print("❌ ДОСТУП ЗАПРЕЩЕН")
    print("=" * 60)

    # Сохраняем результат
    output_path = image_path.replace('.jpg', '_auth_result.jpg')
    cv2.imwrite(output_path, img_result)

    # Показываем результат
    cv2.imshow('Authentication Result', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        'success': True,
        'authenticated': authenticated,
        'similarity': similarity,
        'max_error': max_error,
        'avg_error': avg_error,
        'tolerance': tolerance,
        'common_edges': len(common_edges),
        'output_image': output_path,
        'message': f"Аутентификация: {status_text}. Максимальная ошибка: {max_error:.4f}"
    }


def main():
    """Главная функция"""
    print("=" * 60)
    print("СИСТЕМА АУТЕНТИФИКАЦИИ ПО ЛИЦУ")
    print("=" * 60)

    # Параметры
    user_data_file = "user_data/user_user1_face_features.json"
    test_image = "face_test.jpg"  # Измените на путь к вашему тестовому изображению
    tolerance = 0.05  # Порог ошибки (5% отклонения)

    # Проверяем существование файла с данными
    if not os.path.exists(user_data_file):
        print(f"\n❌ ОШИБКА: Файл {user_data_file} не найден!")
        print("Сначала запустите программу для регистрации пользователя.")
        return

    print(f"\n📁 Данные пользователя: {user_data_file}")
    print(f"📸 Тестовое изображение: {test_image}")
    print(f"🎯 Порог ошибки: {tolerance * 100}%")
    print("\n🔍 Проверка аутентификации...")
    print("-" * 60)

    # Загружаем данные пользователя для вывода информации
    with open(user_data_file, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    print(f"\n👤 Пользователь: {user_data['user_id']}")
    print(f"📅 Дата регистрации: {user_data['timestamp']}")
    print(f"📊 Количество эталонных ребер: {user_data['num_edges']}")
    print(f"📏 Эталонное ребро: {user_data['reference_edge']}")

    # Выполняем аутентификацию
    result = authenticate_face(test_image, user_data_file, tolerance)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ АУТЕНТИФИКАЦИИ")
    print("=" * 60)

    if not result['success']:
        print(f"\n❌ ОШИБКА: {result.get('error', 'Неизвестная ошибка')}")
        return

    print(f"\n📊 Статистика сравнения:")
    print(f"  • Общих ребер: {result['common_edges']}")
    print(f"  • Максимальная ошибка: {result['max_error']:.6f}")
    print(f"  • Средняя ошибка: {result['avg_error']:.6f}")
    print(f"  • Схожесть: {result['similarity'] * 100:.2f}%")
    print(f"  • Порог: {result['tolerance']:.6f}")

    print(f"\n{'=' * 60}")
    if result['authenticated']:
        print("✅ РЕЗУЛЬТАТ: УСПЕШНО - Доступ разрешен")
    else:
        print("❌ РЕЗУЛЬТАТ: НЕУСПЕШНО - Доступ запрещен")
    print("=" * 60)

    print(f"\n📸 Результат сохранен в: {result['output_image']}")
    print("\nНажмите любую клавишу для закрытия окна...")


if __name__ == "__main__":
    import numpy as np

    main()