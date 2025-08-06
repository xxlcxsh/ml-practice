1. Breast cancer classification rating:
    1. Random forest classifier
    best params (max_depth = 8, n_estimators = 100)
    accuracy_score: 0.9912280701754386
    | Класс | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0     | 1.00      | 0.98   | 0.99     | 41      |
    | 1     | 0.99      | 1.00   | 0.99     | 73      |
    Confusion Matrix
    |                 | Предсказано: 0 | Предсказано: 1 |
    |-----------------|----------------|----------------|
    | **Факт: 0**     | 40             | 1              |
    | **Факт: 1**     | 0              | 73             |
    2. Logistic regression classifier
    Accuracy score: 0.9824561403508771
    | Класс         | Precision | Recall | F1-score | Support |
    |---------------|-----------|--------|----------|---------|
    | 0             | 1.00      | 0.95   | 0.97     | 41      |
    | 1             | 0.97      | 1.00   | 0.99     | 73      |
    | **Accuracy**  |           |        | 0.98     | 114     |
    Confusion Matrix:
    |                 | Предсказано: 0 | Предсказано: 1 |
    |-----------------|----------------|----------------|
    | **Факт: 0**     | 39             | 2              |
    | **Факт: 1**     | 0              | 73             |
    3. Gradient boosting classifier
    best params (learning_rate = 0.1, max_depth = 2, n_estimators = 75)
    accuracy score: 9824561403508771
    | Класс         | Precision | Recall | F1-score | Support |
    |---------------|-----------|--------|----------|---------|
    | 0             | 1.00      | 0.95   | 0.97     | 41      |
    | 1             | 0.97      | 1.00   | 0.99     | 73      |
    | **Accuracy**  |           |        | 0.98     | 114     |
    Confusion matrix:
    |                 | Предсказано: 0 | Предсказано: 1 |
    |-----------------|----------------|----------------|
    | **Факт: 0**     | 39             | 2              |
    | **Факт: 1**     | 0              | 73             |
    4. KNN classifier
    best params (n_neighbors= 8)
    Accuracy score: 0.9649122807017544
    | Класс         | Precision | Recall | F1-score | Support |
    |---------------|-----------|--------|----------|---------|
    | 0             | 0.97      | 0.93   | 0.95     | 41      |
    | 1             | 0.96      | 0.99   | 0.97     | 73      |
    | **Accuracy**  |           |        | 0.9649   | 114     |
    Confusion matrix:
    |                 | Предсказано: 0 | Предсказано: 1 |
    |-----------------|----------------|----------------|
    | **Факт: 0**     | 38             | 3              |
    | **Факт: 1**     | 1              | 72             |
    5. Desicion tree classifier
    best_params (max_depth=4)
    Accuracy score: 0.956140350877193
    | Класс         | Precision | Recall | F1-score | Support |
    |---------------|-----------|--------|----------|---------|
    | 0             | 0.97      | 0.90   | 0.94     | 41      |
    | 1             | 0.95      | 0.99   | 0.97     | 73      |
    | **Accuracy**  |           |        | 0.9561   | 114     |
    Confusion Matrix:
    |                 | Предсказано: 0 | Предсказано: 1 |
    |-----------------|----------------|----------------|
    | **Факт: 0**     | 37             | 4              |
    | **Факт: 1**     | 1              | 72             |
2. Titanic classification rating
    1. Logistic regression classifier
    Accuracy: 0.935
    best params (max_iter=10_000)
    report:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    |   0   |   0.96    |  0.94  |   0.95   |   266   |
    |   1   |   0.90    |  0.93  |   0.91   |   152   |
    Confusion matrix:
    |        | Predicted 0 | Predicted 1 |
    |--------|-------------|-------------|
    | True 0 |     250     |     16      |
    | True 1 |     11      |     141     |
    2. Random forest classifier
    Accuracy: 0.906
    Report:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    |   0   |   0.89    |  0.95  |   0.92   |   266   |
    |   1   |   0.90    |  0.80  |   0.85   |   152   |
    Confusion matrix:
    |        | Predicted 0 | Predicted 1 |
    |--------|-------------|-------------|
    | True 0 |     253     |     13      |
    | True 1 |     30      |     122     |
    3. Gradient boosting classifier
    Accuracy: 0.897
    Report:
    | Class | Precision | Recall | F1-score | Support |
    | ----- | --------- | ------ | -------- | ------- |
    | 0     | 0.90      | 0.94   | 0.92     | 266     |
    | 1     | 0.89      | 0.82   | 0.85     | 152     |
    Confusion matrix:
    |          | Predicted 0 | Predicted 1 |
    | -------- | ----------- | ----------- |
    | Actual 0 | 251         | 15          |
    | Actual 1 | 28          | 124         |
    4. KNN classifier
    Accuracy: 0.864
    Report:
    | Класс | Precision | Recall | F1-score | Support |
    | ----- | --------- | ------ | -------- | ------- |
    | 0     | 0.88      | 0.91   | 0.90     | 266     |
    | 1     | 0.84      | 0.78   | 0.81     | 152     |
    Confuison matrix:
    |          | Predicted 0 | Predicted 1 |
    | -------- | ----------- | ----------- |
    | Actual 0 | 243         | 23          |
    | Actual 1 | 34          | 118         |

    5. Decision tree classifier
    Accuracy: 0.813
    Report:
    | Class | Precision | Recall | F1-score | Support |
    | ----- | --------- | ------ | -------- | ------- |
    | 0     | 0.87      | 0.83   | 0.85     | 266     |
    | 1     | 0.73      | 0.78   | 0.75     | 152     |
    Confusion matrix:
    |          | Predicted 0 | Predicted 1 |
    | -------- | ----------- | ----------- |
    | Actual 0 | 221         | 45          |
    | Actual 1 | 33          | 119         |
3. Kaggle spaceship classification rating
    1. Catboost:
    1. Параметры: 
        depth:[3,5,7],iterations:[50,100],learning_rate:[0.05,0.1]
        Лучшие параметры:{'depth': 7, 'iterations': 100, 'learning_rate': 0.05}
        Оценки на каждом фолде: [0.79528465 0.786659   0.78378378 0.80552359 0.80494822]
        Среднее качество: 0.7952398472463111
    2. Параметры:
        'depth': [6,7,8],'iterations':[100,125,150],'learning_rate':[0.05,0.03],
        Лучшие параметры:{'depth': 7, 'iterations': 150, 'learning_rate': 0.05}
        Оценки на каждом фолде: [0.79413456 0.78780909 0.78838413 0.80494822 0.80264672]
        Среднее качество: 0.7955845422583909
    2. Random Forest Classifier
    Параметры: 
    {'clf__max_depth': 11, 'clf__n_estimators': 100} Показатели: 0.7407137152087327
    Оценки на каждом фолде: [0.72742956 0.73375503 0.73433007 0.75201381 0.75143843]
    Среднее качество: 0.7397933815116685