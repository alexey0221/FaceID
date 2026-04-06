import cv2
import numpy as np
import math
import json
import os
import sys
from datetime import datetime
import re
from pathlib import Path
from PIL import Image

class FaceBiometricSystem:
    """Биометрическая система аутентификации по лицу"""

    def __init__(self, data_dir="user_data", models_dir="models"):
        """
        Инициализация системы

        Args:
            data_dir: директория для хранения данных пользователей
            models_dir: директория с моделями
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.face_cascade = None
        self.facemark = None
        self.tolerance = 0.1  # Порог ошибки по умолчанию (10%)

        # Создаем директории если их нет
        os.makedirs(data_dir, exist_ok=True)

        # Загружаем модели
        self._load_models()

    def _load_models(self):
        """Загружает модели OpenCV"""
        cascade_path = 'haarcascade_frontalface_default.xml'
        model_path = "lbfmodel.yaml"

        # Проверяем наличие файлов
        if not os.path.exists(cascade_path):
            print(f"❌ Ошибка: Файл {cascade_path} не найден!")
            return False

        if not os.path.exists(model_path):
            print(f"❌ Ошибка: Файл {model_path} не найден!")
            return False

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.facemark = cv2.face.createFacemarkLBF()

        self.facemark.loadModel(model_path)
        # if not self.facemark.loadModel(model_path):
        #     print("❌ Ошибка: Не удалось загрузить модель LBF")
        #     return False

        print("✅ Модели успешно загружены")
        return True

    def _calculate_distance(self, point1, point2):
        """Вычисляет евклидово расстояние между двумя точками"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _expand_bbox(self, x, y, w, h, image_shape, margin_percent=0.3):
        """Расширяет bounding box с учетом полей"""
        margin_w = int(w * margin_percent)
        margin_h = int(h * margin_percent)

        new_x = max(0, x - margin_w)
        new_y = max(0, y - margin_h)
        new_w = min(image_shape[1] - new_x, w + 2 * margin_w)
        new_h = min(image_shape[0] - new_y, h + 2 * margin_h)

        return new_x, new_y, new_w, new_h

    def _build_custom_face_graph(self, landmarks):
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
            (27, 28), (28, 29), (29, 30), (30, 33)
        ])

        # Горизонтальная линия носа (ноздри 31 и 33 к центру 32)
        connections.extend([
            (31, 32), (32, 33), (33, 34), (34, 35),
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

        # ==================== 6. СВЯЗИ: ВНУТРЕННИЕ УГЛЫ ГЛАЗ К ПЕРЕНОСИЦЕ ====================
        connections.extend([
            (39, 27),  # Левый глаз (внутренний верхний) к переносице
            (42, 27),  # Правый глаз (внутренний верхний) к переносице
        ])

        # ==================== 7. СВЯЗИ: ВНЕШНИЕ УГЛЫ ГЛАЗ К КОНТУРУ ЛИЦА ====================
        connections.extend([
            (36, 0),  # Левый глаз (внешний угол) к левой верхней точке контура
            (45, 16),  # Правый глаз (верхний внешний) к правой верхней точке контура
        ])

        # ==================== 9. СВЯЗИ: НОЗДРИ К КОНТУРУ ЛИЦА ====================
        connections.extend([
            (31, 48),
            (35, 54),
        ])

        # ==================== 10. СВЯЗИ: УГОЛКИ ГУБ К КОНТУРУ ЛИЦА ====================
        connections.extend([
            (48, 4),  # Левый угол рта к нижней левой точке контура
            (54, 12),  # Правый угол рта к нижней правой точке контура
            (57, 8),  # Центр нижней губы к подбородку
        ])

        # ==================== ДОПОЛНИТЕЛЬНЫЕ СВЯЗИ ДЛЯ УЛУЧШЕНИЯ ====================
        # Связи бровей с контуром лица
        connections.extend([
            (17, 0),  # Левая бровь (внешний край) к контуру
            (26, 16),
        ])

        # Связи носа с бровями
        connections.extend([
            (27, 21),  # Переносица к левой брови
            (27, 22),  # Переносица к правой брови
        ])

        connections.extend([
            (30, 0),
            (16,30),
            (35, 14),
            (31, 2),
            (19, 37),
            (24, 44),
        ])

        # Удаляем дубликаты
        connections = list(set(connections))

        # Создаем граф с весами
        n_points = len(points)
        graph = {i: [] for i in range(n_points)}
        edges = []

        for i, j in connections:
            if i < n_points and j < n_points:
                dist = self._calculate_distance(points[i], points[j])
                graph[i].append((j, dist))
                graph[j].append((i, dist))
                edges.append((i, j, dist))

        return edges #graph,

    def _extract_edge_features(self, landmarks, edges, reference_edge=None):#(39, 4)):
        """
        Извлекает характеристики ребер

        Args:
            landmarks: точки лица
            edges: список ребер
            reference_edge: эталонное ребро (если None - используется самое длинное)
        """
        # Если эталонное ребро не указано, находим самое длинное
        if reference_edge is None:
            max_length = 0
            reference_edge = None

            for i, j, dist in edges:
                if dist > max_length:
                    max_length = dist
                    reference_edge = (i, j)

            if reference_edge is None:
                ref_length = 1.0
            else:
                ref_length = max_length

            print(f"Выбрано эталонное ребро: {reference_edge} (длина: {ref_length:.2f} px)")
        else:
            # Находим длину указанного эталонного ребра
            ref_length = None
            for i, j, dist in edges:
                if (i == reference_edge[0] and j == reference_edge[1]) or \
                        (i == reference_edge[1] and j == reference_edge[0]):
                    ref_length = dist
                    break

            if ref_length is None:
                print(f"⚠️ Предупреждение: эталонное ребро {reference_edge} не найдено, используется первое ребро")
                ref_length = edges[0][2] if edges else 1.0

        # Создаем словарь с данными
        edge_features = {}

        for idx, (i, j, abs_length) in enumerate(edges):
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

        return edge_features, ref_length, reference_edge

    def _process_face(self, image):
        """Обрабатывает изображение и возвращает точки лица"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None, None

        # Берем первое лицо
        x, y, w, h = faces[0]
        h_img, w_img = image.shape[:2]

        # Расширяем bounding box
        new_x, new_y, new_w, new_h = self._expand_bbox(x, y, w, h, (h_img, w_img), 0.3)

        # Получаем точки лица
        faces_array = np.array([[x, y, w, h]])
        _, landmarks_result = self.facemark.fit(image, faces_array)

        if not landmarks_result:
            return None, None

        landmarks = []
        for landmark in landmarks_result:
            for (point_x, point_y) in landmark[0]:
                landmarks.append((float(point_x), float(point_y)))

        return landmarks, (new_x, new_y, new_w, new_h)

    def _load_image_pil(self, image_path):
        """
        Загрузка изображения через PIL (лучше работает с путями)
        """
        try:
            # PIL лучше обрабатывает различные пути и кодировки
            img = Image.open(image_path)
            # Конвертируем в формат BGR для совместимости с OpenCV
            img = img.convert('RGB')
            img_np = np.array(img)
            # Конвертируем RGB в BGR (так как OpenCV использует BGR)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            print(f"Ошибка загрузки изображения через PIL: {e}")
            return None

    def register_user(self, username, image_source="camera", image_path=None):
        """
        Регистрация нового пользователя

        Args:
            username: имя пользователя
            image_source: источник изображения ("camera" или "file")
            image_path: путь к файлу (если source="file")

        Returns:
            bool: успешность регистрации
        """
        print(f"\n🔐 Регистрация пользователя: {username}")
        print("-" * 50)

        # Получаем изображение
        image = None

        if image_source == "camera":
            print("📸 Запуск камеры для регистрации...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("❌ Ошибка: Не удалось открыть камеру")
                return False

            print("Нажмите 'SPACE' для захвата изображения, 'ESC' для выхода")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Показываем кадр
                cv2.imshow('Register Face - Press SPACE to capture', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    image = frame.copy()
                    break
                elif key == 27:  # ESC
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif image_source == "file":
            if not image_path or not os.path.exists(image_path):
                print(f"❌ Ошибка: Файл {image_path} не найден")
                return False

            print(f"📁 Загрузка изображения из файла: {image_path}")
            # image_path = re.sub(r"\\", "/", image_path)
            # image_path = re.sub(r"/", '\', image_path)
            # image_path = Path(image_path)
            normalized_path = str(Path(image_path).resolve())
            image = self._load_image_pil(normalized_path)#cv2.imread(normalized_path)

            if image is None:
                print("❌ Ошибка: Не удалось получить изображение")
                return False

        # Обрабатываем лицо
        print("🔍 Обработка изображения...")
        landmarks, bbox = self._process_face(image)

        if landmarks is None or len(landmarks) < 68:
            print("❌ Ошибка: Не удалось обнаружить лицо или недостаточно точек")
            return False

        print(f"✅ Обнаружено {len(landmarks)} точек")

        # Строим граф
        edges = self._build_custom_face_graph(landmarks)

        if not edges:
            print("❌ Ошибка: Не удалось построить граф лица")
            return False

        # Извлекаем характеристики
        edge_features, ref_length, reference_edge = self._extract_edge_features(landmarks, edges)

        print(f"✅ Построен граф из {len(edges)} ребер")
        print(f"📏 Длина эталонного ребра: {ref_length:.2f} px")

        # Сохраняем данные пользователя
        user_data = {
            'user_id': username,
            'timestamp': datetime.now().isoformat(),
            'num_edges': len(edge_features),
            'reference_edge': reference_edge,
            'edges': edge_features,
            'landmarks': landmarks
        }

        filename = os.path.join(self.data_dir, f'user_{username}_face_features.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Пользователь {username} успешно зарегистрирован!")
        print(f"📁 Данные сохранены в: {filename}")

        return True

    def authenticate_user(self, username, image_source="camera", image_path=None, show_resalt = True):
        """
        Аутентификация пользователя

        Args:
            username: имя пользователя
            image_source: источник изображения ("camera" или "file")
            image_path: путь к файлу (если source="file")

        Returns:
            dict: результаты аутентификации
        """
        print(f"\nАутентификация пользователя: {username}")
        print("-" * 50)

        # Загружаем данные пользователя
        user_file = os.path.join(self.data_dir, f'user_{username}_face_features.json')

        if not os.path.exists(user_file):
            print(f"Ошибка: Пользователь {username} не найден в системе")
            return {'authenticated': False, 'error': 'User not found'}

        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)

        print(f"Дата регистрации: {user_data['timestamp']}")
        print(f"Количество эталонных ребер: {user_data['num_edges']}")

        # Получаем изображение
        image = None

        if image_source == "camera":
            print("Запуск камеры для аутентификации...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Ошибка: Не удалось открыть камеру")
                return {'authenticated': False, 'error': 'Camera error'}

            print("Нажмите 'SPACE' для захвата изображения, 'ESC' для выхода")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                cv2.imshow('Authenticate Face - Press SPACE to capture', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    image = frame.copy()
                    break
                elif key == 27:  # ESC
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif image_source == "file":
            if not image_path or not os.path.exists(image_path):
                print(f"Ошибка: Файл {image_path} не найден")
                return {'authenticated': False, 'error': 'File not found'}

            print(f"Загрузка изображения из файла: {image_path}")
            normalized_path = str(Path(image_path).resolve())
            image = self._load_image_pil(normalized_path)#cv2.imread(image_path)

        if image is None:
            print("Ошибка: Не удалось получить изображение")
            return {'authenticated': False, 'error': 'No image'}

        # Обрабатываем лицо
        print("Обработка изображения...")
        landmarks, bbox = self._process_face(image)

        if landmarks is None or len(landmarks) < 68:
            print("Ошибка: Не удалось обнаружить лицо или недостаточно точек")
            return {'authenticated': False, 'error': 'Face not detected'}

        # Строим граф
        edges = self._build_custom_face_graph(landmarks)

        if not edges:
            print("Ошибка: Не удалось построить граф лица")
            return {'authenticated': False, 'error': 'Graph build failed'}

        # Извлекаем характеристики
        current_features, _, reference_edge = self._extract_edge_features(landmarks, edges)

        # Сравниваем с эталоном
        stored_edges = user_data['edges']
        common_edges = set(current_features.keys()) & set(stored_edges.keys())

        if len(common_edges) == 0:
            print("Нет общих ребер для сравнения")
            return {'authenticated': False, 'error': 'No common edges'}

        # Вычисляем ошибки
        errors = []
        for edge_key in common_edges:
            stored_rel_len = stored_edges[edge_key]['relative_length']
            current_rel_len = current_features[edge_key]['relative_length']
            error = abs(stored_rel_len - current_rel_len)
            errors.append(error)

        max_error = max(errors)
        avg_error = np.mean(errors)
        similarity = 1 - min(avg_error / self.tolerance, 1)
        authenticated = avg_error < self.tolerance #max_error < self.tolerance #

        # Визуализация
        img_result = image.copy()

        # Рисуем точки
        for (x, y) in landmarks:
            cv2.circle(img_result, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Рисуем ребра с цветовой индикацией
        for edge_key in common_edges:
            edge_data = stored_edges[edge_key]
            point1_idx = edge_data['point1']
            point2_idx = edge_data['point2']
            error = edge_errors[edge_key] if 'edge_errors' in locals() else 0

            point1 = (int(landmarks[point1_idx][0]), int(landmarks[point1_idx][1]))
            point2 = (int(landmarks[point2_idx][0]), int(landmarks[point2_idx][1]))

            if error < self.tolerance * 0.3:
                color = (0, 255, 0)  # Зеленый
            elif error < self.tolerance * 0.7:
                color = (0, 255, 255)  # Желтый
            else:
                color = (0, 0, 255)  # Красный

            cv2.line(img_result, point1, point2, color, 1)

        # Выводим результат
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТ АУТЕНТИФИКАЦИИ")
        print("=" * 60)

        if authenticated:
            print(f"\nАутентификация: УСПЕШНО")
        else:
            print(f"\nАутентификация: НЕУСПЕШНО")

        print(f"\nДетальная статистика:")
        print(f"  • Максимальная ошибка: {max_error:.4f}")
        print(f"  • Средняя ошибка: {avg_error:.4f}")
        print(f"  • Порог: {self.tolerance:.4f}")
        print(f"  • Схожесть: {similarity * 100:.1f}%")
        print(f"  • Общих ребер: {len(common_edges)}")

        print("\n" + "=" * 60)
        if authenticated:
            print("ДОСТУП РАЗРЕШЕН")
        else:
            print("ДОСТУП ЗАПРЕЩЕН")
        print("=" * 60)

        # Сохраняем результат
        if image_source == "file":
            output_path = image_path.replace('.jpg', '_auth_result.jpg')
        else:
            output_path = f'auth_result_{username}.jpg'

        cv2.imwrite(output_path, img_result)
        print(f"\nРезультат сохранен в: {output_path}")
        if show_resalt:
            cv2.imshow('Authentication Result', img_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            'authenticated': authenticated,
            'similarity': similarity,
            'max_error': max_error,
            'avg_error': avg_error,
            'common_edges': len(common_edges)
        }

    def list_users(self):
        """Выводит список зарегистрированных пользователей"""
        users = []
        for file in os.listdir(self.data_dir):
            if file.startswith('user_') and file.endswith('_face_features.json'):
                username = file[5:-19]  # Убираем "user_" и "_face_features.json"
                users.append(username)
        return users


def main():
    """Главная функция системы"""
    print("=" * 70)
    print("        БИОМЕТРИЧЕСКАЯ СИСТЕМА АУТЕНТИФИКАЦИИ ПО ЛИЦУ")
    print("=" * 70)

    # Инициализация системы
    system = FaceBiometricSystem()

    while True:
        print("\n" + "=" * 70)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 70)
        print("1. Регистрация нового пользователя")
        print("2. Аутентификация пользователя")
        print("3. Список зарегистрированных пользователей")
        print("4. Настройки (порог ошибки)")
        print("5. Выход")
        print("=" * 70)

        choice = input("\nВыберите режим работы (1-5): ").strip()

        if choice == '1':  # Регистрация
            print("\n" + "-" * 50)
            username = input("Введите имя пользователя: ").strip()

            if not username:
                print("❌ Ошибка: Имя пользователя не может быть пустым")
                continue

            # Проверяем, существует ли пользователь
            if username in system.list_users():
                print(f"❌ Ошибка: Пользователь {username} уже существует")
                continue

            print("\nВыберите источник изображения:")
            print("1. С камеры (real-time)")
            print("2. Из файла")
            source_choice = input("Ваш выбор (1-2): ").strip()

            if source_choice == '1':
                system.register_user(username, "camera")
            elif source_choice == '2':
                image_path = input("Введите путь к файлу с изображением: ").strip()
                system.register_user(username, "file", image_path)
            else:
                print("❌ Неверный выбор")

        elif choice == '2':  # Аутентификация
            users = system.list_users()

            if not users:
                print("\n❌ В системе нет зарегистрированных пользователей")
                print("Сначала зарегистрируйте пользователя")
                continue

            print("\nЗарегистрированные пользователи:")
            for i, user in enumerate(users, 1):
                print(f"  {i}. {user}")

            username = input("\nВведите имя пользователя: ").strip()

            if username not in users:
                print(f"❌ Ошибка: Пользователь {username} не найден")
                continue

            print("\nВыберите источник изображения:")
            print("1. С камеры (real-time)")
            print("2. Из файла")
            source_choice = input("Ваш выбор (1-2): ").strip()

            if source_choice == '1':
                system.authenticate_user(username, "camera")
            elif source_choice == '2':
                image_path = input("Введите путь к файлу с изображением: ").strip()
                system.authenticate_user(username, "file", image_path)
            else:
                print("❌ Неверный выбор")

        elif choice == '3':  # Список пользователей
            users = system.list_users()

            if not users:
                print("\n❌ В системе нет зарегистрированных пользователей")
            else:
                print("\n📋 ЗАРЕГИСТРИРОВАННЫЕ ПОЛЬЗОВАТЕЛИ:")
                print("-" * 50)
                for i, user in enumerate(users, 1):
                    # Загружаем информацию о пользователе
                    user_file = os.path.join(system.data_dir, f'user_{user}_face_features.json')
                    with open(user_file, 'r', encoding='utf-8') as f:
                        user_data = json.load(f)

                    print(f"{i}. {user}")
                    print(f"   📅 Дата регистрации: {user_data['timestamp'][:19]}")
                    print(f"   📊 Количество ребер: {user_data['num_edges']}")
                    print()

        elif choice == '4':  # Настройки
            print(f"\n⚙️ Текущий порог ошибки: {system.tolerance * 100:.1f}%")
            new_tolerance = input("Введите новый порог ошибки (в процентах, от 1 до 20): ").strip()

            try:
                new_tolerance = float(new_tolerance) / 100
                if 0.01 <= new_tolerance <= 0.2:
                    system.tolerance = new_tolerance
                    print(f"✅ Порог ошибки установлен на {new_tolerance * 100:.1f}%")
                else:
                    print("❌ Ошибка: Порог должен быть от 1% до 20%")
            except:
                print("❌ Ошибка: Некорректное значение")

        elif choice == '5':  # Выход
            print("\n👋 До свидания!")
            break

        else:
            print("❌ Неверный выбор. Пожалуйста, выберите 1-5")


if __name__ == "__main__":
    import numpy as np

    main()