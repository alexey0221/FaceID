import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from itertools import combinations
import math


def calculate_distance(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


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

def is_intersect(line1, line2):
    """Проверяет, пересекаются ли два отрезка"""

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0
        return 1 if val > 0 else 2

    p1, q1 = line1
    p2, q2 = line2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Проверка на коллинеарность и перекрытие
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False


def on_segment(p, q, r):
    """Проверяет, лежит ли точка q на отрезке pr"""
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False


def build_face_graph(landmarks, max_edge_length_ratio=0.3, k_neighbors=5):
    """
    Строит взвешенный неориентированный граф по точкам на лице

    Args:
        landmarks: список точек (x, y) на лице
        max_edge_length_ratio: максимальная длина ребра относительно размера лица (0.3 = 30%)
        k_neighbors: количество ближайших соседей для соединения

    Returns:
        graph: словарь {индекс_точки: [(сосед, вес), ...]}
        edges: список ребер [(i, j, вес), ...]
    """
    if not landmarks:
        return None, None

    n_points = len(landmarks)
    points = np.array(landmarks)

    # Вычисляем размер лица (максимальное расстояние между точками)
    face_size = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = calculate_distance(points[i], points[j])
            if dist > face_size:
                face_size = dist

    max_edge_length = face_size * max_edge_length_ratio

    # Находим k ближайших соседей для каждой точки
    graph = {i: [] for i in range(n_points)}
    edges = []

    for i in range(n_points):
        # Вычисляем расстояния до всех других точек
        distances = []
        for j in range(n_points):
            if i != j:
                dist = calculate_distance(points[i], points[j])
                if dist <= max_edge_length:
                    distances.append((j, dist))

        # Сортируем по расстоянию и берем k ближайших
        distances.sort(key=lambda x: x[1])
        for j, dist in distances[:k_neighbors]:
            # Проверяем, не добавлено ли уже ребро
            if j not in [neighbor for neighbor, _ in graph[i]]:
                # Проверяем пересечения с существующими ребрами
                new_edge = (points[i], points[j])
                no_intersection = True

                for existing_edge in edges:
                    edge_points = (points[existing_edge[0]], points[existing_edge[1]])
                    if is_intersect(new_edge, edge_points):
                        # Если ребра пересекаются, проверяем, не является ли это пересечение в вершине
                        if not (points_equal(new_edge[0], edge_points[0]) or
                                points_equal(new_edge[0], edge_points[1]) or
                                points_equal(new_edge[1], edge_points[0]) or
                                points_equal(new_edge[1], edge_points[1])):
                            no_intersection = False
                            break

                if no_intersection:
                    graph[i].append((j, dist))
                    graph[j].append((i, dist))
                    edges.append((i, j, dist))

    return graph, edges


def points_equal(p1, p2, tolerance=1e-6):
    """Проверяет равенство двух точек с заданной точностью"""
    return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

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

def build_triangulation_graph(landmarks, max_edge_length_ratio=0.4):
    """
    Строит граф на основе триангуляции Делоне (хорошо подходит для лица)

    Args:
        landmarks: список точек (x, y) на лице
        max_edge_length_ratio: максимальная длина ребра относительно размера лица

    Returns:
        graph: словарь {индекс_точки: [(сосед, вес), ...]}
        edges: список ребер [(i, j, вес), ...]
    """
    if len(landmarks) < 3:
        return None, None

    points = np.array(landmarks)

    # Вычисляем размер лица
    face_size = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = calculate_distance(points[i], points[j])
            if dist > face_size:
                face_size = dist

    max_edge_length = face_size * max_edge_length_ratio

    # Выполняем триангуляцию Делоне
    try:
        tri = Delaunay(points)

        graph = {i: [] for i in range(len(points))}
        edges = []

        # Добавляем ребра из триангуляции
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    p1_idx = simplex[i]
                    p2_idx = simplex[j]
                    dist = calculate_distance(points[p1_idx], points[p2_idx])

                    # Проверяем длину ребра
                    if dist <= max_edge_length:
                        if p2_idx not in [neighbor for neighbor, _ in graph[p1_idx]]:
                            graph[p1_idx].append((p2_idx, dist))
                            graph[p2_idx].append((p1_idx, dist))
                            edges.append((p1_idx, p2_idx, dist))

        return graph, edges
    except Exception as e:
        print(f"Ошибка при триангуляции: {e}")
        return None, None


def build_anatomical_graph(landmarks):
    """
    Строит анатомически обоснованный граф на основе известных структур лица
    (соединяет точки в соответствии с анатомией)

    Args:
        landmarks: список точек (x, y) в порядке, соответствующем анатомии

    Returns:
        graph: словарь {индекс_точки: [(сосед, вес), ...]}
        edges: список ребер [(i, j, вес), ...]
    """
    # Стандартные связи для 68 точек (dlib/OpenFace)
    # Это примерные связи для основных контуров лица
    anatomical_connections = [
        # Контур лица
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        # Брови
        (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),  # Левая бровь
        (22, 23), (23, 24), (24, 25), (25, 26),  # Правая бровь
        # Нос
        (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
        # Глаза
        (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # Левый глаз
        (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # Правый глаз
        # Рот
        (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56),
        (56, 57), (57, 58), (58, 59), (59, 48),  # Внешний контур рта
        (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)  # Внутренний контур рта
    ]

    points = np.array(landmarks)
    n_points = len(points)

    graph = {i: [] for i in range(n_points)}
    edges = []

    for i, j in anatomical_connections:
        if i < n_points and j < n_points:
            dist = calculate_distance(points[i], points[j])
            graph[i].append((j, dist))
            graph[j].append((i, dist))
            edges.append((i, j, dist))

    return graph, edges


def visualize_graph(image, landmarks, edges, title="Face Graph"):
    """
    Визуализирует граф на изображении

    Args:
        image: исходное изображение
        landmarks: точки на лице
        edges: список ребер [(i, j, вес), ...]
        title: заголовок окна
    """
    img_copy = image.copy()

    # Рисуем ребра графа
    for i, j, weight in edges:
        point1 = (int(landmarks[i][0]), int(landmarks[i][1]))
        point2 = (int(landmarks[j][0]), int(landmarks[j][1]))

        # Цвет зависит от веса (длины ребра)
        normalized_weight = min(weight / 100.0, 1.0)  # Нормализуем вес
        color = (int(255 * normalized_weight), int(255 * (1 - normalized_weight)), 0)

        cv2.line(img_copy, point1, point2, color, 1)

    # Рисуем точки
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_copy, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow(title, img_copy)


def get_landmarks_from_results(processed_faces, face_index=0):
    """
    Извлекает точки из результатов обработки

    Args:
        processed_faces: результат из process_face
        face_index: индекс лица для обработки

    Returns:
        список точек (x, y)
    """
    if not processed_faces or face_index >= len(processed_faces):
        return None

    face_data = processed_faces[face_index]
    landmarks = []

    if face_data['landmarks']:
        for landmark in face_data['landmarks']:
            for (x, y) in landmark[0]:
                landmarks.append((float(x), float(y)))

    return landmarks


def main_with_graph():
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
        # Получаем точки для первого лица
        landmarks = get_landmarks_from_results(processed_faces, 0)

        if landmarks and len(landmarks) > 0:
            print(f"Получено {len(landmarks)} точек")

            # Строим граф разными способами
            print("\n1. Граф на основе ближайших соседей:")
            graph_knn, edges_knn = build_face_graph(landmarks, max_edge_length_ratio=0.25, k_neighbors=4)
            if edges_knn:
                print(f"  Создано {len(edges_knn)} ребер")
                visualize_graph(result_image, landmarks, edges_knn, "K-Nearest Neighbors Graph")

            print("\n2. Граф на основе триангуляции Делоне:")
            graph_delaunay, edges_delaunay = build_triangulation_graph(landmarks, max_edge_length_ratio=0.35)
            if edges_delaunay:
                print(f"  Создано {len(edges_delaunay)} ребер")
                visualize_graph(result_image, landmarks, edges_delaunay, "Delaunay Triangulation Graph")

            print("\n3. Анатомический граф:")
            graph_anatomical, edges_anatomical = build_anatomical_graph(landmarks)
            if edges_anatomical:
                print(f"  Создано {len(edges_anatomical)} ребер")
                visualize_graph(result_image, landmarks, edges_anatomical, "Anatomical Graph")

            # Выводим информацию о графах
            print("\nСтатистика по графам:")
            for name, edges in [("KNN", edges_knn), ("Delaunay", edges_delaunay), ("Anatomical", edges_anatomical)]:
                if edges:
                    weights = [w for _, _, w in edges]
                    print(f"{name}:")
                    print(f"  - Количество ребер: {len(edges)}")
                    print(f"  - Средний вес: {np.mean(weights):.2f}")
                    print(f"  - Минимальный вес: {np.min(weights):.2f}")
                    print(f"  - Максимальный вес: {np.max(weights):.2f}")

            cv2.imshow('Original with landmarks', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Точки не найдены")
    else:
        print("Не удалось обработать изображение")


if __name__ == "__main__":
    main_with_graph()