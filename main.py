import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def load_sensor_data(filepath):
    df = pd.read_csv(filepath, header=None, delim_whitespace=True)
    df.columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    return df

def extract_features(filepath):
    df = load_sensor_data(filepath)
    n_fft = 64
    fft_vals = np.abs(np.fft.fft(df['Ax'].values[:n_fft]))
    feats = {
        'fft_mean_Ax': np.mean(fft_vals),
        'fft_std_Ax': np.std(fft_vals),
        'fft_max_Ax': np.max(fft_vals),
        'fft_min_Ax': np.min(fft_vals),
        'Ax_mean': df['Ax'].mean(),
        'Ay_mean': df['Ay'].mean(),
        'Az_mean': df['Az'].mean(),
        'Gx_mean': df['Gx'].mean(),
        'Gy_mean': df['Gy'].mean(),
        'Gz_mean': df['Gz'].mean(),
        'Ax_std': df['Ax'].std(),
        'Ay_std': df['Ay'].std(),
        'Az_std': df['Az'].std(),
        'Gx_std': df['Gx'].std(),
        'Gy_std': df['Gy'].std(),
        'Gz_std': df['Gz'].std(),
    }
    return feats

def main():
    train_info = pd.read_csv('train_info.csv')
    print("Extracting features from train data...")
    feature_list = []
    for uid in train_info['unique_id']:
        filepath = os.path.join('train_data', f'{uid}.txt')
        feats = extract_features(filepath)
        feature_list.append(feats)
    X = pd.DataFrame(feature_list)

    target_cols = ['gender', 'hold racket handed', 'play years', 'level']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoders = {}
    y_encoded = {}
    for col in target_cols:
        le = LabelEncoder()
        y_encoded[col] = le.fit_transform(train_info[col])
        encoders[col] = le

    classifiers = {}
    print("Training models...")
    for col in target_cols:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y_encoded[col])
        classifiers[col] = clf

    print("Validation accuracy on training set:")
    for col in target_cols:
        score = classifiers[col].score(X_scaled, y_encoded[col])
        print(f'  {col}: {score:.4f}')

    # 測試集處理
    test_info = pd.read_csv('test_info.csv')
    test_features = []
    print("Extracting features from test data...")
    for uid in test_info['unique_id']:
        filepath = os.path.join('test_data', f'{uid}.txt')
        feats = extract_features(filepath)
        test_features.append(feats)
    X_test = pd.DataFrame(test_features)
    X_test_scaled = scaler.transform(X_test)

    preds = {'unique_id': test_info['unique_id']}

    # 二分類: gender, hold racket handed
    for col in ['gender', 'hold racket handed']:
        clf = classifiers[col]
        le = encoders[col]
        probas = clf.predict_proba(X_test_scaled)
        # 找 label=1 的 index (確保有 1)
        if 1 in le.classes_:
            idx_1 = list(le.classes_).index(1)
            preds[col] = probas[:, idx_1]
        else:
            # 若無 label=1，填 0
            preds[col] = np.zeros(len(test_info))

    # 多分類 play years (0,1,2)
    col = 'play years'
    clf = classifiers[col]
    le = encoders[col]
    probas = clf.predict_proba(X_test_scaled)
    for class_label in [0, 1, 2]:
        if class_label in le.classes_:
            idx = list(le.classes_).index(class_label)
            preds[f'{col}_{class_label}'] = probas[:, idx]
        else:
            preds[f'{col}_{class_label}'] = np.zeros(len(test_info))

    # 多分類 level (2,3,4,5)
    col = 'level'
    clf = classifiers[col]
    le = encoders[col]
    probas = clf.predict_proba(X_test_scaled)
    for class_label in [2, 3, 4, 5]:
        if class_label in le.classes_:
            idx = list(le.classes_).index(class_label)
            preds[f'{col}_{class_label}'] = probas[:, idx]
        else:
            preds[f'{col}_{class_label}'] = np.zeros(len(test_info))

    submission = pd.DataFrame(preds)

    # 四捨五入
    submission = submission.round(4)
    submission.to_csv('sample_submission.csv', index=False)

if __name__ == '__main__':
    main()
