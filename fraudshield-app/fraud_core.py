"""
FRAUD SHIELD – LÕI MÔ HÌNH (dùng chung cho ứng dụng web và chạy dòng lệnh)
Festival Chuyển đổi số 2026 · Bảng B · Digital Payment Fraud Shield

Ba lớp:
  1) Hồ sơ hành vi khách hàng (leave-one-out, không dùng nhãn)
  2) Chấm điểm 3 tầng: Rule (noisy-OR) + Logistic Regression + Isolation Forest
     risk_score = 0,60·ML + 0,30·Rule + 0,10·Anomaly
  3) Ngưỡng / risk_level / alert_reason / priority_rank / gom case

Hai chế độ:
  - "train": Data Pack có fraud_labels_sample.csv -> huấn luyện + đánh giá đầy đủ
    (5-fold phân tầng, ngưỡng lồng, baseline, theo thời gian, top-k, chi phí, công bằng).
  - "score": Data Pack không có nhãn -> chấm điểm bằng mô hình đã huấn luyện sẵn
    (model/pretrained.joblib, học trên 2.000 nhãn công khai của BTC).
"""
import hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
VERSION = "FraudShield-v1.0 (Profile+Rules+LR+IsoForest)"
W_FUSION = (0.60, 0.30, 0.10)
RULE_W = {"flag_R1_strong": .80, "flag_R1_mild": .35, "flag_R2": .80, "flag_R3": .80, "flag_R4": .60,
          "flag_R5": .85, "flag_R6": .85, "flag_C1": .15, "flag_C2": .10}
RULE_NAMES = {"flag_R1_strong": "R1 mạnh – số tiền ≥ 5 lần mức nền",
              "flag_R1_mild": "R1 nhẹ – số tiền 3–5 lần mức nền",
              "flag_R2": "R2 – thiết bị lạ, mới xuất hiện ≤ 30 ngày",
              "flag_R3": "R3 – IP nước ngoài và tỉnh khác nơi ở",
              "flag_R4": "R4 – merchant HIGH kèm tiền ≥ 2× hoặc tỉnh mới",
              "flag_R5": "R5 – chuỗi thất bại rồi thành công trong ±30 phút",
              "flag_R6": "R6 – ≥ 3 giao dịch trong ±5 phút",
              "flag_C1": "C1 – thiết bị root/giả lập/rủi ro HIGH",
              "flag_C2": "C2 – không xác thực"}
CASE_GAP_S = 1800
COST_FN, COST_FP = 10, 1
F_BASE = ["amt_ratio", "dev_not_registered_to_cust", "dev_age_days", "dev_is_new", "loc_mismatch", "ip_foreign",
          "merchant_high", "auth_none", "dev_compromised", "n_txn_5m", "log_min_gap", "n_fail_30m", "n_succ_30m"]
F_PROFILE = ["prof_amt_z", "prof_new_device", "prof_new_location", "prof_new_channel", "prof_new_ip",
             "profile_dev_count"]
FEATURES = F_BASE + F_PROFILE
OUT_COLS = ["transaction_id", "risk_score", "risk_level", "predicted_fraud", "alert_reason", "priority_rank",
            "confidence", "model_name", "processing_time_ms"]

# Cột bắt buộc theo từng tệp (tên tệp theo Data Pack của BTC)
REQUIRED = {
    "transactions.csv": ["transaction_id", "customer_id", "merchant_id", "datetime", "amount", "payment_channel",
                         "device_id", "location", "transaction_status", "ip_country", "authentication_method"],
    "customers.csv": ["customer_id", "customer_segment", "home_location", "avg_transaction_amount"],
    "merchants.csv": ["merchant_id", "merchant_category", "merchant_risk_level"],
    "devices.csv": ["device_id", "first_seen_date", "registered_customer_id", "device_risk_level",
                    "rooted_or_jailbroken", "emulator_flag"],
}
OPTIONAL = {"fraud_labels_sample.csv": ["transaction_id", "is_fraud", "fraud_type"],
            "dataset_manifest.csv": ["file_name", "row_count", "column_count", "sha256"],
            "submission_output_template.csv": ["transaction_id"],
            "answer_key_sample.csv": ["transaction_id", "expected_is_fraud", "expected_risk_level"]}

PRETRAINED = Path(__file__).parent / "model" / "pretrained.json"


# ============================================================================ tiện ích
def vnd(x):
    return f"{int(round(x)):,}".replace(",", ".") + " VND"


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def dot(n):
    return f"{int(n):,}".replace(",", ".")


def _csv(b):
    return pd.read_csv(io.BytesIO(b), encoding="utf-8-sig")


# ============================================================================ 0. ĐỌC FILE TẢI LÊN
def collect_files(items):
    """items: list of (filename, bytes). Nhận .zip (đọc trong bộ nhớ, không giải nén ra đĩa) hoặc .csv.
    Trả về dict tên-tệp (chữ thường) -> bytes."""
    files, notes = {}, []
    for name, data in items:
        low = name.lower()
        if low.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        base = info.filename.replace("\\", "/").split("/")[-1].lower()
                        if info.is_dir() or not base.endswith(".csv") or base.startswith("._"):
                            continue
                        if info.file_size > 200 * 1024 * 1024:
                            notes.append(f"Bỏ qua {base}: quá lớn"); continue
                        files[base] = z.read(info)
            except zipfile.BadZipFile:
                notes.append(f"{name}: không phải tệp zip hợp lệ")
        elif low.endswith(".csv"):
            files[low.split("/")[-1]] = data
        else:
            notes.append(f"Bỏ qua {name}: chỉ nhận .zip hoặc .csv")
    return files, notes


def collect_dir(path):
    files = {}
    for p in Path(path).rglob("*.csv"):
        files[p.name.lower()] = p.read_bytes()
    return files


# ============================================================================ 1. KIỂM TRA DỮ LIỆU
def validate(files):
    """Kiểm tra mềm: trả về (bảng dữ liệu, danh sách kiểm tra, có chạy được không)."""
    checks, T = [], {}

    def add(group, check, status, detail=""):
        checks.append({"group": group, "check": check, "status": status, "detail": detail})

    for name, cols in REQUIRED.items():
        if name not in files:
            add("Tệp bắt buộc", name, "LỖI", "Không tìm thấy tệp này trong dữ liệu tải lên"); continue
        try:
            t = _csv(files[name])
        except Exception as e:  # noqa
            add("Tệp bắt buộc", name, "LỖI", f"Không đọc được CSV: {e}"); continue
        miss = [c for c in cols if c not in t.columns]
        if miss:
            add("Tệp bắt buộc", name, "LỖI", "Thiếu cột: " + ", ".join(miss)); continue
        T[name] = t
        add("Tệp bắt buộc", name, "ĐẠT", f"{dot(len(t))} dòng × {t.shape[1]} cột")
    for name, cols in OPTIONAL.items():
        if name not in files:
            add("Tệp tùy chọn", name, "THIẾU", "Không bắt buộc"); continue
        try:
            t = _csv(files[name])
            miss = [c for c in cols if c not in t.columns]
            if miss:
                add("Tệp tùy chọn", name, "CẢNH BÁO", "Thiếu cột: " + ", ".join(miss) + " – bỏ qua tệp"); continue
            T[name] = t
            add("Tệp tùy chọn", name, "ĐẠT", f"{dot(len(t))} dòng")
        except Exception as e:  # noqa
            add("Tệp tùy chọn", name, "CẢNH BÁO", f"Không đọc được: {e}")

    ok = all(n in T for n in REQUIRED)
    if not ok:
        return T, checks, False

    # Đối chiếu manifest của BTC (nếu có)
    if "dataset_manifest.csv" in T:
        man = T["dataset_manifest.csv"]
        for _, r in man.iterrows():
            fn = str(r.file_name).lower()
            if fn not in files:
                continue
            t = _csv(files[fn]); sha = hashlib.sha256(files[fn]).hexdigest()
            same = (len(t) == r.row_count) and (t.shape[1] == r.column_count) and (sha == r.sha256)
            add("Đối chiếu manifest BTC", fn, "ĐẠT" if same else "CẢNH BÁO",
                "Khớp số dòng, số cột và SHA-256" if same else
                f"Khác manifest (dòng {len(t)}/{r.row_count}, cột {t.shape[1]}/{r.column_count}, "
                f"SHA {'khớp' if sha == r.sha256 else 'khác'}) – vẫn chạy được")

    gd, kh, mc, tb = (T[n] for n in ["transactions.csv", "customers.csv", "merchants.csv", "devices.csv"])
    uniq = gd.transaction_id.is_unique
    add("Chất lượng dữ liệu", "transaction_id không trùng", "ĐẠT" if uniq else "LỖI",
        "" if uniq else f"{gd.transaction_id.duplicated().sum()} mã bị trùng")
    amt = pd.to_numeric(gd.amount, errors="coerce")
    bad_amt = int((amt.isna() | (amt <= 0)).sum())
    add("Chất lượng dữ liệu", "amount là số dương", "ĐẠT" if bad_amt == 0 else "LỖI",
        "" if bad_amt == 0 else f"{bad_amt} dòng không hợp lệ")
    dt = pd.to_datetime(gd.datetime, errors="coerce")
    bad_dt = int(dt.isna().sum())
    add("Chất lượng dữ liệu", "datetime đọc được", "ĐẠT" if bad_dt == 0 else "LỖI",
        f"Từ {dt.min():%d/%m/%Y} đến {dt.max():%d/%m/%Y}" if bad_dt == 0 else f"{bad_dt} dòng lỗi")
    fk_ok = True
    for k, t, nm in [("customer_id", kh, "customers"), ("merchant_id", mc, "merchants"), ("device_id", tb, "devices")]:
        miss = int((~gd[k].isin(t[k])).sum())
        fk_ok &= miss == 0
        add("Liên kết bảng", f"{k} có trong {nm}.csv", "ĐẠT" if miss == 0 else "LỖI",
            "100% giao dịch liên kết được" if miss == 0 else f"{miss} giao dịch không tìm thấy")
    for k, t, nm in [("customer_id", kh, "customers"), ("merchant_id", mc, "merchants"), ("device_id", tb, "devices")]:
        if not t[k].is_unique:
            fk_ok = False
            add("Liên kết bảng", f"{k} không trùng trong {nm}.csv", "LỖI", "Có mã trùng, ghép bảng sẽ nhân dòng")
    if "fraud_labels_sample.csv" in T:
        lb = T["fraud_labels_sample.csv"]
        inn = lb.transaction_id.isin(gd.transaction_id)
        add("Nhãn", "Nhãn khớp giao dịch", "ĐẠT" if inn.all() else "CẢNH BÁO",
            f"{dot(int(inn.sum()))} nhãn, {int(lb.is_fraud.sum())} gian lận ({num(lb.is_fraud.mean() * 100)}%); "
            f"{dot(len(gd) - int(inn.sum()))} giao dịch không nhãn chỉ được chấm điểm, không coi là bình thường")
    ok = uniq and bad_amt == 0 and bad_dt == 0 and fk_ok
    return T, checks, ok


# ============================================================================ 2. FEATURE + HỒ SƠ
def window_count(ts, w):
    return np.searchsorted(ts, ts + w, "right") - np.searchsorted(ts, ts - w, "left")


def build(gd, kh, mc, tb):
    gd = gd.copy(); tb = tb.copy()
    gd["datetime"] = pd.to_datetime(gd.datetime); gd["amount"] = pd.to_numeric(gd.amount)
    tb["first_seen_date"] = pd.to_datetime(tb.first_seen_date)
    df = gd.merge(kh, on="customer_id", how="left").merge(mc, on="merchant_id", how="left") \
           .merge(tb, on="device_id", how="left")
    assert len(df) == len(gd)
    df["amt_ratio"] = df.amount / df.avg_transaction_amount
    df["dev_not_registered_to_cust"] = (df.registered_customer_id != df.customer_id).astype(int)
    df["dev_age_days"] = (df.datetime.dt.normalize() - df.first_seen_date).dt.days.clip(lower=0)
    df["dev_is_new"] = (df.dev_age_days <= 30).astype(int)
    df["loc_mismatch"] = (df.location != df.home_location).astype(int)
    df["ip_foreign"] = (df.ip_country != "VN").astype(int)
    df["merchant_high"] = (df.merchant_risk_level == "HIGH").astype(int)
    df["auth_none"] = (df.authentication_method == "NONE").astype(int)
    tobool = lambda s: s.astype(str).str.upper().isin(["TRUE", "1", "YES"])
    df["dev_compromised"] = ((df.device_risk_level == "HIGH") | tobool(df.rooted_or_jailbroken)
                             | tobool(df.emulator_flag)).astype(int)
    df = df.sort_values(["customer_id", "datetime"]).reset_index(drop=True)
    df["ts"] = (df.datetime - pd.Timestamp("2000-01-01")).dt.total_seconds()
    n5, f30, s30, gap = [], [], [], []
    for _, g in df.groupby("customer_id", sort=False):
        ts = g.ts.values; f = (g.transaction_status != "SUCCESS").astype(int).values
        cf = np.concatenate([[0], np.cumsum(f)])
        lo = np.searchsorted(ts, ts - CASE_GAP_S, "left"); hi = np.searchsorted(ts, ts + CASE_GAP_S, "right")
        n5.append(window_count(ts, 300)); f30.append(cf[hi] - cf[lo]); s30.append((hi - lo) - (cf[hi] - cf[lo]))
        gp = np.diff(ts, prepend=np.nan); gn = np.append(np.diff(ts), np.nan)
        gap.append(np.fmin(gp, gn))
    df["n_txn_5m"] = np.concatenate(n5); df["n_fail_30m"] = np.concatenate(f30)
    df["n_succ_30m"] = np.concatenate(s30)
    df["min_gap_s"] = np.concatenate(gap); df["log_min_gap"] = np.log1p(df.min_gap_s.fillna(1e7))
    for cot, ten in [("device_id", "device"), ("location", "location"), ("payment_channel", "channel"),
                     ("ip_country", "ip")]:
        df[f"prof_new_{ten}"] = (df.groupby(["customer_id", cot])["transaction_id"].transform("count") == 1).astype(int)
    la = np.log(df.amount); k = df.customer_id
    n = la.groupby(k).transform("count"); s1 = la.groupby(k).transform("sum"); s2 = (la ** 2).groupby(k).transform("sum")
    m = ((s1 - la) / (n - 1)).where(n > 1, la)
    sd = np.sqrt(((s2 - la ** 2) / (n - 1).clip(lower=1) - m ** 2).clip(lower=0)).clip(lower=0.3)
    df["prof_amt_z"] = ((la - m) / sd).where(n > 1, 0.0); df["prof_typical_amt"] = np.exp(m)
    df["prof_amt_dev"] = (df.prof_amt_z >= 3).astype(int)
    df["cust_n_txn"] = n.astype(int)
    df["n_same_ip_other"] = df.groupby(["customer_id", "ip_country"])["transaction_id"].transform("count") - 1
    df["profile_dev_count"] = df[["prof_amt_dev", "prof_new_device", "prof_new_location", "prof_new_channel",
                                  "prof_new_ip"]].sum(axis=1)
    return df


def rules(df, use_profile=True):
    f = pd.DataFrame(index=df.index)
    f["flag_R1_strong"] = df.amt_ratio >= 5
    f["flag_R1_mild"] = (df.amt_ratio >= 3) & (df.amt_ratio < 5)
    f["flag_R2"] = (df.dev_not_registered_to_cust == 1) & (df.dev_is_new == 1)
    f["flag_R3"] = (df.ip_foreign == 1) & (df.loc_mismatch == 1)
    f["flag_R4"] = (df.merchant_high == 1) & ((df.amt_ratio >= 2) | ((df.prof_new_location == 1) & use_profile))
    f["flag_R5"] = (df.n_fail_30m >= 2) & (df.n_succ_30m >= 1)
    f["flag_R6"] = df.n_txn_5m >= 3
    f["flag_C1"] = df.dev_compromised == 1
    f["flag_C2"] = df.auth_none == 1
    f = f.astype(int)
    score = 1 - np.prod([1 - RULE_W[c] * f[c].values for c in RULE_W], axis=0)
    return f, score


def reason(r):
    p = []
    if r.flag_R5: p.append(f"Thuộc chuỗi {r.n_fail_30m} lần FAILED/DECLINED và {r.n_succ_30m} lần SUCCESS của cùng khách hàng trong ±30 phút")
    if r.flag_R6: p.append(f"{r.n_txn_5m} giao dịch của cùng khách hàng trong ±5 phút")
    if r.flag_R1_strong or r.flag_R1_mild:
        p.append(f"Số tiền {vnd(r.amount)} gấp {num(r.amt_ratio)} lần mức nền {vnd(r.avg_transaction_amount)}")
    if r.flag_R2: p.append(f"Thiết bị {r.device_id} chưa đăng ký cho khách hàng, mới xuất hiện {int(r.dev_age_days)} ngày")
    if r.flag_R3: p.append(f"Giao dịch tại {r.location} (nơi ở: {r.home_location}) với IP {r.ip_country}; "
                           f"trong {r.cust_n_txn - 1} giao dịch khác của khách, {r.n_same_ip_other} giao dịch dùng IP {r.ip_country}")
    if r.flag_R4:
        if r.flag_R1_strong or r.flag_R1_mild:
            p.append(f"Merchant {r.merchant_id} ({r.merchant_category}) có mức rủi ro nền HIGH")
        else:
            x = f"số tiền gấp {num(r.amt_ratio)} lần mức nền" if r.amt_ratio >= 2 else f"khách lần đầu giao dịch tại {r.location}"
            p.append(f"Merchant {r.merchant_id} ({r.merchant_category}) rủi ro nền HIGH, {x}")
    prof = []
    if r.prof_amt_dev and not (r.flag_R1_strong or r.flag_R1_mild):
        prof.append(f"số tiền {vnd(r.amount)} so với mức thường {vnd(r.prof_typical_amt)}")
    if r.prof_new_device and not r.flag_R2: prof.append(f"lần đầu dùng thiết bị {r.device_id}")
    if r.prof_new_location and not (r.flag_R3 or r.flag_R4): prof.append(f"lần đầu giao dịch tại {r.location}")
    if r.prof_new_ip and not r.flag_R3: prof.append(f"lần đầu dùng IP {r.ip_country}")
    if prof: p.append("Lệch hồ sơ hành vi: " + ", ".join(prof))
    if r.flag_C1: p.append(f"Thiết bị {r.device_id} có cờ rủi ro (root/emulator/rủi ro nền HIGH)")
    if r.flag_C2: p.append("Giao dịch không xác thực (NONE)")
    if not p:
        return "Không có tín hiệu bất thường nổi bật so với hồ sơ nền."
    return "; ".join(p[:3]) + "."


# ============================================================================ 3. MÔ HÌNH + ĐÁNH GIÁ
def best_thr(y, s):
    p, r, t = precision_recall_curve(y, s); f = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f[:-1])])


def evaluate(y, s, folds):
    pred = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        pred[va] = s[va] >= best_thr(y[tr], s[tr])
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "pr_auc": average_precision_score(y, s), "fpr": fp / int((y == 0).sum()),
            "tp": tp, "fp": fp, "fn": fn, "pred": pred}


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def oof(X, y, folds):
    o = np.zeros(len(y))
    for tr, va in folds:
        o[va] = lr().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return o


def iso_score(X):
    s = -IsolationForest(n_estimators=300, random_state=SEED).fit(X).score_samples(X)
    return pd.Series(s).rank(pct=True).values


def r4(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def run(files, tables=None, progress=lambda msg: None, pretrained_path=PRETRAINED):
    """Chạy toàn bộ Fraud Shield. Trả về dict kết quả (DataFrame + báo cáo)."""
    if tables is None:
        tables, checks, ok = validate(files)
        if not ok:
            raise ValueError("Dữ liệu chưa đạt kiểm tra bắt buộc")
    gd, kh, mc, tb = (tables[n] for n in ["transactions.csv", "customers.csv", "merchants.csv", "devices.csv"])
    nhan = tables.get("fraud_labels_sample.csv")
    t0 = time.perf_counter()
    progress("Đang tạo feature và hồ sơ hành vi…")
    df = build(gd, kh, mc, tb)
    flags_B, df["rule_score"] = rules(df, True)
    _, rule_A = rules(df, False)
    df = df.join(flags_B)
    X_all = df[FEATURES].fillna(0).values
    progress("Đang chấm điểm bất thường (Isolation Forest)…")
    df["anomaly_score"] = iso_score(X_all)
    feat_ms = (time.perf_counter() - t0) * 1000
    w = W_FUSION
    rep = {"version": VERSION, "seed": SEED, "fusion_weights": W_FUSION, "rule_weights": RULE_W,
           "features": FEATURES, "n_transactions": int(len(df))}
    lab_view = None

    labeled = nhan is not None
    if labeled:
        nhan = nhan[nhan.transaction_id.isin(df.transaction_id)].drop_duplicates("transaction_id")
        yy = nhan.is_fraud.astype(int)
        labeled = len(nhan) >= 50 and yy.sum() >= 10 and (yy == 0).sum() >= 10
    if labeled:
        mode = "train"
        progress("Đang huấn luyện và đánh giá (5-fold, ngưỡng lồng)…")
        anom_A = iso_score(df[F_BASE].fillna(0).values)
        idx = df.index[df.transaction_id.isin(nhan.transaction_id)].values
        lab = df.loc[idx].merge(nhan[["transaction_id", "is_fraud", "fraud_type"]], on="transaction_id",
                                how="left").set_index(idx)
        y = lab.is_fraud.values.astype(int); ft = lab.fraud_type.fillna("NORMAL").values
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
        ml_B = oof(lab[FEATURES].fillna(0).values, y, folds)
        ml_A = oof(lab[F_BASE].fillna(0).values, y, folds)
        rB, rA = df.loc[idx, "rule_score"].values, rule_A[idx]
        aB, aA = df.loc[idx, "anomaly_score"].values, anom_A[idx]
        cands = {"Chỉ Rule": rB, "Chỉ Isolation Forest": aB, "Chỉ Logistic Regression": ml_B,
                 "Gộp 3 tầng – không hồ sơ (A)": w[0] * ml_A + w[1] * rA + w[2] * aA,
                 "Gộp 3 tầng – có hồ sơ (B, chính thức)": w[0] * ml_B + w[1] * rB + w[2] * aB}
        comp, per_type = {}, {}
        for k, s in cands.items():
            e = evaluate(y, s, folds); per_type[k] = pd.Series(e.pop("pred")).groupby(ft).sum().astype(int).to_dict()
            comp[k] = r4(e)
        fused = cands["Gộp 3 tầng – có hồ sơ (B, chính thức)"]
        counts = pd.Series(ft).value_counts().to_dict()
        order = np.argsort(-fused)
        topk = [{"k": k, "fraud": int(y[order[:k]].sum()), "precision": round(y[order[:k]].sum() / k, 4),
                 "recall": round(y[order[:k]].sum() / y.sum(), 4)} for k in (25, 50, 100, 150, 200) if k <= len(y)]
        grid = np.unique(np.round(fused, 4))
        costs = [COST_FN * ((fused < g) & (y == 1)).sum() + COST_FP * ((fused >= g) & (y == 0)).sum() for g in grid]
        cthr = float(grid[int(np.argmin(costs))]); cp = fused >= cthr
        cost = {"threshold": round(cthr, 4), "recall": round(recall_score(y, cp), 4),
                "precision": round(precision_score(y, cp), 4), "alerts_labeled": int(cp.sum()),
                "fn": int(((~cp) & (y == 1)).sum()), "fp": int((cp & (y == 0)).sum()),
                "detected_by_type": pd.Series(cp.astype(int)).groupby(ft).sum().astype(int).to_dict()}
        Xl = lab[FEATURES].fillna(0).values
        tsplit = None
        try:
            mth = lab.datetime.dt.to_period("M"); months = sorted(mth.unique())
            cut = months[int(len(months) * 2 / 3) - 1]
            trm, tem = (mth <= cut).values, (mth > cut).values
            if y[trm].sum() >= 10 and y[tem].sum() >= 5:
                f2 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(trm.sum()), y[trm]))
                s_tr = w[0] * oof(Xl[trm], y[trm], f2) + w[1] * rB[trm] + w[2] * aB[trm]
                thr_t = best_thr(y[trm], s_tr)
                s_te = w[0] * lr().fit(Xl[trm], y[trm]).predict_proba(Xl[tem])[:, 1] + w[1] * rB[tem] + w[2] * aB[tem]
                pt = s_te >= thr_t
                tsplit = {"train_months": f"{months[0]} → {cut}", "test_months": f"{months[months.index(cut) + 1]} → {months[-1]}",
                          "n_train": int(trm.sum()), "fraud_train": int(y[trm].sum()), "n_test": int(tem.sum()),
                          "fraud_test": int(y[tem].sum()), "precision": round(precision_score(y[tem], pt), 4),
                          "recall": round(recall_score(y[tem], pt), 4), "f1": round(f1_score(y[tem], pt), 4),
                          "pr_auc": round(average_precision_score(y[tem], s_te), 4),
                          "fp": int((pt & (y[tem] == 0)).sum()), "fn": int((~pt & (y[tem] == 1)).sum())}
        except Exception:  # noqa – dữ liệu quá ít tháng
            tsplit = None
        THR = best_thr(y, fused)
        model = lr().fit(Xl, y)
        ml_all = model.predict_proba(X_all)[:, 1]; ml_all[idx] = ml_B
        labf = lab.assign(pred=evaluate(y, fused, folds)["pred"], risk_oof=fused)
        fpr_seg = labf[labf.is_fraud == 0].groupby("customer_segment").pred.mean().round(4).to_dict()
        err_cols = ["transaction_id", "fraud_type", "amount", "avg_transaction_amount", "amt_ratio", "location",
                    "home_location", "ip_country", "merchant_risk_level", "profile_dev_count", "risk_oof"]
        rep.update({"labeled_counts": {k: int(v) for k, v in counts.items()}, "comparison_nested_cv": comp,
                    "detected_by_type": per_type, "topk": topk, "cost_threshold": cost, "time_split": tsplit,
                    "fpr_by_segment_labeled": fpr_seg,
                    "false_negatives": labf[(labf.is_fraud == 1) & (labf.pred == 0)][err_cols].round(3).to_dict("records"),
                    "false_positives": labf[(labf.is_fraud == 0) & (labf.pred == 1)][err_cols].round(3).to_dict("records")})
        lab_view = pd.DataFrame({"transaction_id": lab.transaction_id.values, "y": y, "fraud_type": ft,
                                 "score": np.round(fused, 5)})
        model_info = {"source": "Huấn luyện trên nhãn trong Data Pack vừa tải lên", "n_labels": int(len(y)),
                      "n_fraud": int(y.sum())}
    else:
        mode = "score"
        progress("Không có nhãn – chấm điểm bằng mô hình đã huấn luyện sẵn…")
        pack = json.loads(Path(pretrained_path).read_text(encoding="utf-8"))
        THR = pack["threshold"]
        z = ((X_all - np.array(pack["mean"])) / np.array(pack["scale"])) @ np.array(pack["coef"]) + pack["intercept"]
        ml_all = 1 / (1 + np.exp(-z))
        model_info = {"source": pack.get("source", "Mô hình huấn luyện sẵn"), "n_labels": pack.get("n_labels"),
                      "n_fraud": pack.get("n_fraud")}

    progress("Đang xếp hạng, viết lý do và gom case…")
    t1 = time.perf_counter()
    df["ml_score"] = ml_all
    df["risk_score"] = (w[0] * df.ml_score + w[1] * df.rule_score + w[2] * df.anomaly_score).clip(0, 1)
    df["predicted_fraud"] = (df.risk_score >= THR).astype(int)
    crit = (df.flag_R5 == 1) | (df.flag_R6 == 1)
    lvl = np.where(df.risk_score >= THR, "HIGH", np.where(df.risk_score >= 0.6 * THR, "MEDIUM", "LOW"))
    df["risk_level"] = np.where((df.predicted_fraud == 1) & crit, "CRITICAL", lvl)
    d = np.where(df.predicted_fraud == 1, (df.risk_score - THR) / (1 - THR), (THR - df.risk_score) / THR)
    df["confidence"] = (0.5 + 0.5 * d).clip(0, 1).round(4)
    df["alert_reason"] = [reason(r) for r in df.itertuples()]
    df = df.sort_values(["risk_score", "amount"], ascending=[False, False]).reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)
    infer_ms = (time.perf_counter() - t1) * 1000 + feat_ms
    df["processing_time_ms"] = round(infer_ms / len(df), 3)
    df["model_name"] = VERSION
    df["risk_score"] = df.risk_score.round(4)

    cases = make_cases(df)
    df["case_id"] = df.transaction_id.map(cases.attrs["map"])

    rep.update({"mode": mode, "model_info": model_info, "threshold": round(THR, 4),
                "n_alerts": int(df.predicted_fraud.sum()), "n_cases": int(len(cases)),
                "case_reduction": round(1 - len(cases) / max(1, df.predicted_fraud.sum()), 4),
                "risk_level_counts": {k: int(v) for k, v in df.risk_level.value_counts().to_dict().items()},
                "alert_rate_by_segment": df.groupby("customer_segment").predicted_fraud.mean().round(4).to_dict(),
                "alert_rate_by_home_location": df.groupby("home_location").predicted_fraud.mean().round(4).to_dict(),
                "rule_hits_on_alerts": {RULE_NAMES[c]: int(df.loc[df.predicted_fraud == 1, c].sum()) for c in RULE_W},
                "latency_ms_per_txn_batch": round(infer_ms / len(df), 3),
                "period": [f"{df.datetime.min():%d/%m/%Y}", f"{df.datetime.max():%d/%m/%Y}"]})
    if "answer_key_sample.csv" in tables:
        ak = tables["answer_key_sample.csv"].merge(df[["transaction_id", "predicted_fraud", "risk_level"]],
                                                   on="transaction_id", how="inner")
        if len(ak):
            fr = ak[ak.expected_is_fraud == 1]
            rep["answer_key"] = {"n": int(len(ak)), "agree_alert": int((ak.expected_is_fraud == ak.predicted_fraud).sum()),
                                 "fraud_alerted": int(fr.predicted_fraud.sum()), "n_fraud": int(len(fr)),
                                 "level_match_on_alerted": int((fr.expected_risk_level == fr.risk_level)[fr.predicted_fraud == 1].sum()),
                                 "normal_alerted": int(ak[ak.expected_is_fraud == 0].predicted_fraud.sum()),
                                 "n_normal": int((ak.expected_is_fraud == 0).sum())}
    # File nộp theo template
    if "submission_output_template.csv" in tables:
        tmpl = tables["submission_output_template.csv"]
        cols = [c for c in tmpl.columns if c in df.columns] or OUT_COLS
        sub = tmpl[["transaction_id"]].merge(df[cols], on="transaction_id", how="left")
    else:
        order = {t: i for i, t in enumerate(gd.transaction_id)}
        sub = df[OUT_COLS].sort_values("transaction_id", key=lambda s: s.map(order)).reset_index(drop=True)
    return {"df": df, "cases": cases, "report": rep, "submission": sub, "lab_view": lab_view, "mode": mode}


def make_cases(df):
    a = df[df.predicted_fraud == 1].sort_values(["customer_id", "datetime"]).copy()
    rank_lvl = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    if a.empty:
        c = pd.DataFrame(columns=["case_id", "case_rank", "customer_id", "n_transactions", "start_time", "end_time",
                                  "total_amount", "max_risk_score", "case_level", "devices", "transaction_ids",
                                  "top_reason"])
        c.attrs["map"] = {}
        return c
    new = (a.customer_id != a.customer_id.shift()) | (a.datetime.diff().dt.total_seconds() > CASE_GAP_S)
    a["seq"] = new.cumsum()
    top = a.sort_values("risk_score", ascending=False).drop_duplicates("seq").set_index("seq").alert_reason
    cases = a.groupby("seq").agg(customer_id=("customer_id", "first"), n_transactions=("transaction_id", "size"),
                                 start_time=("datetime", "min"), end_time=("datetime", "max"),
                                 total_amount=("amount", "sum"), max_risk_score=("risk_score", "max"),
                                 case_level=("risk_level", lambda s: max(s, key=rank_lvl.get)),
                                 devices=("device_id", lambda s: ";".join(sorted(set(s)))),
                                 transaction_ids=("transaction_id", lambda s: ";".join(s))).reset_index()
    cases["top_reason"] = cases.seq.map(top)
    cases = cases.sort_values(["max_risk_score", "n_transactions"], ascending=[False, False]).reset_index(drop=True)
    cases.insert(0, "case_rank", np.arange(1, len(cases) + 1))
    cases.insert(0, "case_id", [f"CASE{i:04d}" for i in cases.case_rank])
    m = dict(zip(a.transaction_id, a.seq.map(dict(zip(cases.seq, cases.case_id)))))
    cases = cases.drop(columns="seq")
    cases.attrs["map"] = m
    return cases


def train_pretrained(files, out_path=PRETRAINED):
    """Huấn luyện mô hình dùng cho chế độ chỉ chấm điểm (chạy một lần trên Data Pack có nhãn)."""
    res = run(files)
    assert res["mode"] == "train"
    df, lv = res["df"], res["lab_view"]
    lab = df[df.transaction_id.isin(lv.transaction_id)].merge(lv[["transaction_id", "y"]], on="transaction_id")
    model = lr().fit(lab[FEATURES].fillna(0).values, lab.y.values)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sc, clf = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    pack = {"version": VERSION, "features": FEATURES, "threshold": res["report"]["threshold"],
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]), "n_labels": int(len(lab)), "n_fraud": int(lab.y.sum()),
            "source": "Học trên 2.000 nhãn công khai của Data Pack BTC (snapshot 18/10/2026)"}
    Path(out_path).write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    import sys
    res = run(collect_dir(sys.argv[1]))
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    res["submission"].to_csv(out / "submission_output.csv", index=False, encoding="utf-8-sig")
    res["cases"].to_csv(out / "cases.csv", index=False, encoding="utf-8-sig")
    (out / "kpi_report.json").write_text(json.dumps(res["report"], ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    print(json.dumps({k: res["report"].get(k) for k in ["mode", "threshold", "n_alerts", "n_cases", "risk_level_counts"]},
                     ensure_ascii=False))
