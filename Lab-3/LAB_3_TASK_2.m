% Створення нечіткої системи керування кондиціонером
fis = mamfis('Name', 'AirConditioningControl');

% Додавання вхідної змінної "Температура"
fis = addInput(fis, [10 40], 'Name', 'Temperature');
fis = addMF(fis, 'Temperature', 'trapmf', [10 10 18 20], 'Name', 'VeryCold');
fis = addMF(fis, 'Temperature', 'trapmf', [18 20 24 26], 'Name', 'Cold');
fis = addMF(fis, 'Temperature', 'trimf', [24 25 26], 'Name', 'Normal');
fis = addMF(fis, 'Temperature', 'trapmf', [26 28 32 34], 'Name', 'Warm');
fis = addMF(fis, 'Temperature', 'trapmf', [32 34 40 40], 'Name', 'VeryWarm');

% Додавання вхідної змінної "Швидкість зміни температури"
fis = addInput(fis, [-5 5], 'Name', 'TemperatureChangeRate');
fis = addMF(fis, 'TemperatureChangeRate', 'trapmf', [-5 -5 -2 0], 'Name', 'Negative');
fis = addMF(fis, 'TemperatureChangeRate', 'trimf', [-0.5 0 0.5], 'Name', 'Zero');
fis = addMF(fis, 'TemperatureChangeRate', 'trapmf', [0 2 5 5], 'Name', 'Positive');

% Додавання вихідної змінної "Режим кондиціонера"
fis = addOutput(fis, [-10 10], 'Name', 'ACMode');
fis = addMF(fis, 'ACMode', 'trapmf', [-10 -8 -6 -4], 'Name', 'StrongCool');
fis = addMF(fis, 'ACMode', 'trimf', [-6 -4 -2], 'Name', 'WeakCool');
fis = addMF(fis, 'ACMode', 'trimf', [-1 0 1], 'Name', 'Off');
fis = addMF(fis, 'ACMode', 'trimf', [2 4 6], 'Name', 'WeakHeat');
fis = addMF(fis, 'ACMode', 'trapmf', [6 8 10 10], 'Name', 'StrongHeat');

% Додавання нечітких правил
ruleList = [
    1 3 1 1 1;   % Якщо температура дуже тепла і швидкість додатня, то сильний холод
    1 1 2 1 1;   % Якщо температура дуже тепла і швидкість від'ємна, то слабкий холод
    4 3 1 1 1;   % Якщо температура тепла і швидкість додатня, то сильний холод
    4 1 3 1 1;   % Якщо температура тепла і швидкість від'ємна, то вимкнути
    2 1 5 1 1;   % Якщо температура дуже холодна і швидкість від'ємна, то сильне тепло
    2 3 4 1 1;   % Якщо температура дуже холодна і швидкість додатня, то слабке тепло
    3 1 5 1 1;   % Якщо температура холодна і швидкість від'ємна, то сильне тепло
    3 3 3 1 1;   % Якщо температура холодна і швидкість додатня, то вимкнути
    1 2 1 1 1;   % Якщо температура дуже тепла і швидкість 0, то сильний холод
    4 2 2 1 1;   % Якщо температура тепла і швидкість 0, то слабкий холод
    2 2 5 1 1;   % Якщо температура дуже холодна і швидкість 0, то сильне тепло
    3 2 4 1 1;   % Якщо температура холодна і швидкість 0, то слабке тепло
    5 3 2 1 1;   % Якщо температура в нормі і швидкість додатня, то слабкий холод
    5 1 4 1 1;   % Якщо температура в нормі і швидкість від'ємна, то слабке тепло
    5 2 3 1 1;   % Якщо температура в нормі і швидкість 0, то вимкнути
];

% Додавання правил до нечіткої системи
fis = addRule(fis, ruleList);

% Виведення отриманої нечіткої системи
ruleview(fis);    % Візуалізація правил
surfview(fis);    % Візуалізація поверхні керування

% Можна також зберегти отриману нечітку систему у файл
writefis(fis, 'AirConditioningControl');
