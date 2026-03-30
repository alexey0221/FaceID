import cv2
import numpy as np
import math
import json
import os
from datetime import datetime


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
        print(f"Лицо {i + 1}: исходный bounding box: x={x}, y={y}, w={w}, h={h}")

        new_x, new_y, new_w, new_h = expand_bbox(x, y, w, h, (h_img, w_img), margin_percent)
        print(f"Лицо {i + 1}: расширенный bounding box: x={new_x}, y={new_y}, w={new_w}, h={new_h}")

        face_roi = img_copy[new_y:new_y + new_h, new_x:new_x + new_w]

        faces_array = np.array([[x, y, w, h]])
        _, landmarks = facemark.fit(img_copy, faces_array)

        results.append({
            'bbox': (x, y, w, h),
            'expanded_bbox': (new_x, new_y, new_w, new_h),
            'face_roi': face_roi,
            'landmarks': landmarks
        })

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(img_copy, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

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
        print(f"Предупреждение: получено {len(landmarks) if landmarks else 0} точек, ожидается 68")
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
    graph = {i: [] for i in range(n_points)}
    edges = []

    for i, j in connections:
        if i < n_points and j < n_points:
            dist = calculate_distance(points[i], points[j])
            graph[i].append((j, dist))
            graph[j].append((i, dist))
            edges.append((i, j, dist))

    return graph, edges


def visualize_graph(image, landmarks, edges, title="Face Graph"):
    """Визуализирует граф на изображении"""
    img_copy = image.copy()

    # Рисуем ребра графа
    for i, j, weight in edges:
        point1 = (int(landmarks[i][0]), int(landmarks[i][1]))
        point2 = (int(landmarks[j][0]), int(landmarks[j][1]))

        # Цвет ребра зависит от типа (синий для основных, зеленый для дополнительных)
        cv2.line(img_copy, point1, point2, (255, 100, 0), 1)

    # Рисуем точки
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_copy, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imshow(title, img_copy)
    return img_copy


def extract_edge_features(landmarks, edges, reference_edge=(39, 4)):
    """
    Извлекает характеристики ребер: абсолютные длины и относительные (относительно эталонного ребра)

    Args:
        landmarks: список точек
        edges: список ребер [(i, j, dist), ...]
        reference_edge: эталонное ребро (индексы точек)

    Returns:
        dict: словарь с данными о ребрах
    """
    # Находим длину эталонного ребра
    ref_length = None
    for i, j, dist in edges:
        if (i == reference_edge[0] and j == reference_edge[1]) or \
                (i == reference_edge[1] and j == reference_edge[0]):
            ref_length = dist
            break

    if ref_length is None:
        print(f"Предупреждение: эталонное ребро {reference_edge} не найдено")
        ref_length = 1.0

    # Создаем словарь с данными
    edge_features = {}

    for idx, (i, j, abs_length) in enumerate(edges):
        # Создаем уникальный ключ для ребра (упорядочиваем индексы)
        edge_key = f"{min(i, j)}_{max(i, j)}"
        relative_length = abs_length / ref_length

        edge_features[edge_key] = {
            'edge_id': idx,
            'point1': i,
            'point2': j,
            'absolute_length_px': abs_length,
            'relative_length': relative_length,
            'is_reference': (i == reference_edge[0] and j == reference_edge[1]) or \
                            (i == reference_edge[1] and j == reference_edge[0])
        }

    return edge_features, ref_length


def save_user_data(user_id, edge_features, landmarks, output_dir="user_data"):
    """
    Сохраняет данные пользователя в файл

    Args:
        user_id: идентификатор пользователя
        edge_features: словарь с характеристиками ребер
        landmarks: точки лица
        output_dir: директория для сохранения
    """
    # Создаем директорию, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Подготавливаем данные для сохранения
    user_data = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'num_edges': len(edge_features),
        'reference_edge': '39_4',
        'edges': edge_features,
        'landmarks': landmarks  # Сохраняем для отладки
    }

    # Сохраняем в JSON файл
    filename = os.path.join(output_dir, f'user_{user_id}_face_features.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    print(f"Данные пользователя {user_id} сохранены в {filename}")

    # Также сохраняем в CSV для удобства анализа
    csv_filename = os.path.join(output_dir, f'user_{user_id}_edges.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write("edge_id,point1,point2,absolute_length_px,relative_length,is_reference\n")
        for edge_key, data in edge_features.items():
            f.write(f"{data['edge_id']},{data['point1']},{data['point2']},"
                    f"{data['absolute_length_px']:.2f},{data['relative_length']:.6f},"
                    f"{data['is_reference']}\n")

    print(f"CSV файл сохранен в {csv_filename}")

    return filename


def load_user_data(user_id, output_dir="user_data"):
    """
    Загружает данные пользователя из файла

    Args:
        user_id: идентификатор пользователя
        output_dir: директория с данными

    Returns:
        dict: данные пользователя или None
    """
    filename = os.path.join(output_dir, f'user_{user_id}_face_features.json')

    if not os.path.exists(filename):
        print(f"Файл {filename} не найден")
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    return user_data


def compare_faces(user_data1, user_data2, tolerance=0.05):
    """
    Сравнивает два набора данных для аутентификации

    Args:
        user_data1: данные первого пользователя
        user_data2: данные второго пользователя
        tolerance: допустимое отклонение относительных длин

    Returns:
        dict: результаты сравнения
    """
    edges1 = user_data1['edges']
    edges2 = user_data2['edges']

    # Находим общие ребра
    common_edges = set(edges1.keys()) & set(edges2.keys())

    if len(common_edges) == 0:
        return {
            'match': False,
            'similarity': 0,
            'message': "Нет общих ребер для сравнения"
        }

    differences = []
    for edge_key in common_edges:
        rel_len1 = edges1[edge_key]['relative_length']
        rel_len2 = edges2[edge_key]['relative_length']
        diff = abs(rel_len1 - rel_len2)
        differences.append(diff)

    avg_diff = np.mean(differences)
    similarity = 1 - min(avg_diff / tolerance, 1)
    is_match = avg_diff < tolerance

    return {
        'match': is_match,
        'similarity': similarity,
        'avg_difference': avg_diff,
        'common_edges': len(common_edges),
        'tolerance': tolerance,
        'message': f"Схожесть: {similarity * 100:.2f}%, Среднее отклонение: {avg_diff:.4f}"
    }


def main():
    # Загрузка моделей
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("lbfmodel.yaml")

    # Загрузка изображения
    image = cv2.imread('face.jpg')
    if image is None:
        print("Ошибка: Не удалось загрузить изображение")
        return

    # Обработка изображения
    MARGIN_PERCENT = 0.3
    result_image, processed_faces = process_face(image, face_cascade, facemark, MARGIN_PERCENT)

    if result_image is not None and processed_faces:
        landmarks = get_landmarks_from_results(processed_faces, 0)

        if landmarks and len(landmarks) > 0:
            print(f"Получено {len(landmarks)} точек")

            # Строим пользовательский граф
            print("\nПользовательский граф лица:")
            graph_custom, edges_custom = build_custom_face_graph(landmarks)

            if edges_custom:
                print(f"  Создано {len(edges_custom)} ребер")

                # Извлекаем характеристики ребер
                reference_edge = (39, 4)  # Эталонное ребро
                edge_features, ref_length = extract_edge_features(landmarks, edges_custom, reference_edge)

                print(f"\nДлина эталонного ребра (39-4): {ref_length:.2f} пикселей")

                # Выводим информацию о первых 10 ребрах
                print("\nПервые 10 ребер:")
                print("=" * 80)
                print(f"{'ID':<6} {'Точки':<10} {'Абс. длина':<12} {'Отн. длина':<12} {'Эталон':<6}")
                print("-" * 80)

                for i, (edge_key, data) in enumerate(list(edge_features.items())[:10]):
                    print(f"{data['edge_id']:<6} {data['point1']}-{data['point2']:<6} "
                          f"{data['absolute_length_px']:<12.2f} {data['relative_length']:<12.6f} "
                          f"{'Да' if data['is_reference'] else 'Нет':<6}")

                # Сохраняем данные пользователя
                user_id = "user1"
                save_user_data(user_id, edge_features, landmarks)

                # Демонстрация загрузки и сравнения (для теста загружаем те же данные)
                print(f"\nЗагружаем данные пользователя {user_id} для проверки...")
                loaded_data = load_user_data(user_id)

                if loaded_data:
                    print(f"Данные загружены: пользователь {loaded_data['user_id']}")
                    print(f"Количество ребер: {loaded_data['num_edges']}")
                    print(f"Временная метка: {loaded_data['timestamp']}")

                    # Сравниваем с самим собой (должно показать высокую схожесть)
                    comparison = compare_faces(loaded_data, loaded_data, tolerance=0.05)
                    print(f"\nРезультат аутентификации (сравнение с самим собой):")
                    print(f"  {comparison['message']}")
                    print(f"  {'✅ ДОСТУП РАЗРЕШЕН' if comparison['match'] else '❌ ДОСТУП ЗАПРЕЩЕН'}")

                # Визуализируем граф
                img_result = visualize_graph(result_image, landmarks, edges_custom,
                                             "Custom Face Graph")
                cv2.imwrite('face_graph_custom.jpg', img_result)

                # Сохраняем обрезанное лицо с графом
                for face_data in processed_faces:
                    face_with_points = face_data['face_roi'].copy()
                    new_x, new_y, new_w, new_h = face_data['expanded_bbox']

                    for i, j, _ in edges_custom:
                        point1_x = int(landmarks[i][0] - new_x)
                        point1_y = int(landmarks[i][1] - new_y)
                        point2_x = int(landmarks[j][0] - new_x)
                        point2_y = int(landmarks[j][1] - new_y)

                        if (0 <= point1_x < new_w and 0 <= point1_y < new_h and
                                0 <= point2_x < new_w and 0 <= point2_y < new_h):
                            cv2.line(face_with_points, (point1_x, point1_y),
                                     (point2_x, point2_y), (255, 100, 0), 2)

                    for (x, y) in landmarks:
                        adj_x = int(x - new_x)
                        adj_y = int(y - new_y)
                        if 0 <= adj_x < new_w and 0 <= adj_y < new_h:
                            cv2.circle(face_with_points, (adj_x, adj_y), 3, (0, 255, 0), -1)

                    cv2.imwrite(f'cropped_face_with_custom_graph.jpg', face_with_points)

                # Статистика
                weights = [w for _, _, w in edges_custom]
                print(f"\nСтатистика графа:")
                print(f"  - Количество ребер: {len(edges_custom)}")
                print(f"  - Средняя длина: {np.mean(weights):.2f} px")
                print(f"  - Минимальная длина: {np.min(weights):.2f} px")
                print(f"  - Максимальная длина: {np.max(weights):.2f} px")
                print(f"  - Эталонное ребро: {ref_length:.2f} px")

            # cv2.imshow('Original with landmarks', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Точки не найдены")
    else:
        print("Не удалось обработать изображение")


def get_landmarks_from_results(processed_faces, face_index=0):
    """Извлекает точки из результатов обработки"""
    if not processed_faces or face_index >= len(processed_faces):
        return None

    face_data = processed_faces[face_index]
    landmarks = []

    if face_data['landmarks']:
        for landmark in face_data['landmarks']:
            for (x, y) in landmark[0]:
                landmarks.append((float(x), float(y)))

    return landmarks


if __name__ == "__main__":
    import numpy as np

    main()