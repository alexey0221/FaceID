import cv2
import numpy as np
import math


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

    Args:
        landmarks: список точек (x, y) на лице

    Returns:
        graph: словарь {индекс_точки: [(сосед, вес), ...]}
        edges: список ребер [(i, j, вес), ...]
    """
    if not landmarks or len(landmarks) < 68:
        print(f"Предупреждение: получено {len(landmarks) if landmarks else 0} точек, ожидается 68")
        return None, None

    points = np.array(landmarks)
    connections = []

    # ==================== 1. КОНТУР ЛИЦА ====================
    # Соединяем каждую соседнюю точку контура (0-16)
    for i in range(16):
        connections.append((i, i + 1))

    # ==================== 2. БРОВИ ====================
    # Левая бровь (17-21) - соединяем соседние
    for i in range(17, 21):
        connections.append((i, i + 1))

    # Правая бровь (22-26) - соединяем соседние
    for i in range(22, 26):
        connections.append((i, i + 1))

    # ==================== 3. НОС (перевернутая буква Т) ====================
    # Вертикальная линия носа (27-28-29-30-32)
    connections.extend([
        (27, 28), (28, 29), (29, 30), (30, 32)
    ])

    # Горизонтальная линия носа (ноздри 31 и 33 к центру 32)
    connections.extend([
        (31, 32), (32, 33)
    ])

    # ==================== 4. ГЛАЗА ====================
    # Левая глаз (36-41) - обводка
    connections.extend([
        (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)
    ])

    # Правая глаз (42-47) - обводка
    connections.extend([
        (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)
    ])

    # ==================== 5. ГУБЫ ====================
    # Внешний контур губ (48-59)
    for i in range(48, 59):
        connections.append((i, i + 1))
    connections.append((59, 48))  # Замыкаем внешний контур

    # Внутренний контур губ (60-67)
    for i in range(60, 67):
        connections.append((i, i + 1))
    connections.append((67, 60))  # Замыкаем внутренний контур

    # ==================== 6. СВЯЗИ: ВНУТРЕННИЕ УГЛЫ ГЛАЗ К ПЕРЕНОСИЦЕ ====================
    connections.extend([
        (39, 27),  # Левый глаз (внутренний верхний) к переносице
        (40, 27),  # Левый глаз (внутренний нижний) к переносице
        (42, 27),  # Правый глаз (внутренний верхний) к переносице
        (47, 27),  # Правый глаз (внутренний нижний) к переносице
    ])

    # ==================== 7. СВЯЗИ: ВНЕШНИЕ УГЛЫ ГЛАЗ К КОНТУРУ ЛИЦА ====================
    connections.extend([
        (36, 0),  # Левый глаз (внешний угол) к левой верхней точке контура
        (36, 1),  # Левый глаз (внешний угол) к левой верхней точке контура
        (36, 2),  # Левый глаз (внешний угол) к левой верхней точке контура
        (36, 3),  # Левый глаз (внешний угол) к левой верхней точке контура
        (38, 3),  # Левый глаз (внешний угол) к левой верхней точке контура
        (39, 4),  # Левый глаз (нижний внешний) к левой средней точке контура
        (45, 16),  # Правый глаз (верхний внешний) к правой верхней точке контура
        (45, 15),  # Правый глаз (верхний внешний) к правой верхней точке контура
        (46, 15),  # Правый глаз (нижний внешний) к правой средней точке контура
        (46, 14),  # Правый глаз (нижний внешний) к правой средней точке контура
        (46, 13),  # Правый глаз (нижний внешний) к правой средней точке контура
    ])

    # ==================== 8. СВЯЗИ: БРОВИ К ВЕРХНИМ ТОЧКАМ ГЛАЗ ====================
    connections.extend([
        (18, 37),  # Левая бровь (внешняя середина) к левому глазу
        (19, 38),  # Левая бровь (центр) к центру левого глаза
        (20, 39),  # Левая бровь (внутренняя середина) к внутреннему углу левого глаза

        (23, 44),  # Правая бровь (внутренняя середина) к центру правого глаза
        (24, 43),  # Правая бровь (центр) к внутреннему углу правого глаза
        (25, 45),  # Правая бровь (внешняя середина) к внешнему углу правого глаза
    ])

    # ==================== 9. СВЯЗИ: НОЗДРИ К КОНТУРУ ЛИЦА ====================
    connections.extend([
        (31, 6),  # Левая ноздря к левой средней точке контура
        (33, 10),  # Правая ноздря к правой средней точке контура
    ])

    # ==================== 10. СВЯЗИ: УГОЛКИ ГУБ К КОНТУРУ ЛИЦА ====================
    connections.extend([
        (48, 4),  # Левый угол рта к нижней левой точке контура
        (48, 5),  # Левый угол рта к нижней левой точке контура
        (50, 7),  # Левый угол рта к нижней левой точке контура
        (54, 11),  # Правый угол рта к нижней правой точке контура
        (54, 12),  # Правый угол рта к нижней правой точке контура
        (54, 13),  # Правый угол рта к нижней правой точке контура
        (57, 10),  # Центр нижней губы к подбородку
        (57, 8),  # Центр нижней губы к подбородку
        (57, 9),  # Центр нижней губы к подбородку
    ])

    # ==================== ДОПОЛНИТЕЛЬНЫЕ СВЯЗИ ДЛЯ УЛУЧШЕНИЯ ====================
    # Связи бровей с контуром лица
    connections.extend([
        (17, 0),  # Левая бровь (внешний край) к контуру
        # (21, 4),  # Левая бровь (внутренний край) к контуру
        (22, 13),  # Правая бровь (внутренний край) к контуру
        # (26, 12),  # Правая бровь (внешний край) к контуру
    ])

    # Связи носа с бровями
    connections.extend([
        (27, 21),  # Переносица к левой брови
        (27, 22),  # Переносица к правой брови
    ])
    # Кончик носа и уголки глаз
    connections.extend([
        (30, 39),
        (30, 42),
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


def lines_intersect(p1, q1, p2, q2):
    """Проверяет, пересекаются ли два отрезка"""

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


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

                # Проверяем количество пересечений
                intersection_count = 0
                for idx1, (i1, j1, _) in enumerate(edges_custom):
                    for idx2, (i2, j2, _) in enumerate(edges_custom):
                        if idx1 < idx2:
                            p1 = (landmarks[i1][0], landmarks[i1][1])
                            q1 = (landmarks[j1][0], landmarks[j1][1])
                            p2 = (landmarks[i2][0], landmarks[i2][1])
                            q2 = (landmarks[j2][0], landmarks[j2][1])

                            if lines_intersect(p1, q1, p2, q2):
                                intersection_count += 1

                print(f"  Количество пересечений: {intersection_count}")

                # Визуализируем
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
                print(f"\nСтатистика:")
                print(f"  - Количество ребер: {len(edges_custom)}")
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
    import numpy as np

    main()