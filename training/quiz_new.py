"""Quick head-to-head on fresh, hand-authored Indonesian questions.

These 10 items are NOT from the training/eval data. Each has one correct
passage (paraphrased, low lexical overlap with the query) plus topical
distractors. We check whether each reranker puts the correct passage first.

Auth: set OPENROUTER_API_KEY. Run:
    python training/quiz_new.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

OPENROUTER_URL = "https://openrouter.ai/api/v1/rerank"

# Each item: query, the correct (paraphrased) passage, and distractors.
QUIZ = [
    {
        "q": "Bagaimana cara menjaga kesehatan jantung?",
        "correct": "Rutin berolahraga dan menghindari makanan berlemak dapat memperkuat organ pemompa darah.",
        "distractors": [
            "Harga tiket pesawat biasanya naik selama musim liburan.",
            "Aplikasi perpesanan baru diluncurkan pekan ini.",
            "Hutan hujan tropis menyimpan keanekaragaman hayati yang tinggi.",
            "Pameran otomotif tahunan menampilkan mobil konsep terbaru.",
        ],
    },
    {
        "q": "Siapa pencipta lagu Indonesia Raya?",
        "correct": "Wage Rudolf Supratman adalah komponis yang menggubah lagu kebangsaan Indonesia.",
        "distractors": [
            "Bunga melati sering digunakan dalam upacara adat.",
            "Tim bulu tangkis meraih medali emas di kejuaraan dunia.",
            "Gunung berapi itu kembali mengeluarkan abu vulkanik.",
            "Sungai terpanjang di Kalimantan adalah Sungai Kapuas.",
        ],
    },
    {
        "q": "Mengapa air laut terasa asin?",
        "correct": "Kandungan garam mineral yang terbawa aliran sungai membuat samudra memiliki rasa asin.",
        "distractors": [
            "Kereta cepat mempersingkat waktu tempuh antarkota.",
            "Festival film tahunan diadakan di ibu kota.",
            "Cuaca cerah diperkirakan berlangsung sepanjang akhir pekan.",
            "Burung-burung bermigrasi saat musim dingin tiba.",
        ],
    },
    {
        "q": "Apa manfaat membaca buku setiap hari?",
        "correct": "Kebiasaan membaca dapat memperluas wawasan dan meningkatkan kemampuan berpikir kritis.",
        "distractors": [
            "Restoran itu terkenal dengan hidangan lautnya.",
            "Pemerintah membangun bendungan baru di daerah itu.",
            "Pertandingan sepak bola berakhir imbang tanpa gol.",
            "Pohon kelapa banyak tumbuh di sepanjang pantai.",
        ],
    },
    {
        "q": "Bagaimana proses terjadinya hujan?",
        "correct": "Uap air yang menguap dari permukaan bumi mengembun menjadi awan lalu jatuh sebagai titik-titik air.",
        "distractors": [
            "Mata uang kripto mengalami fluktuasi harga yang tajam.",
            "Museum sejarah memamerkan artefak kuno.",
            "Petani memanen padi pada musim kemarau.",
            "Konser musik itu dihadiri ribuan penonton.",
        ],
    },
    {
        "q": "Di mana letak Candi Borobudur?",
        "correct": "Monumen Buddha terbesar di dunia itu berada di Magelang, Jawa Tengah.",
        "distractors": [
            "Resep sambal terasi membutuhkan cabai dan bawang.",
            "Perusahaan teknologi merilis ponsel lipat terbarunya.",
            "Ikan paus adalah mamalia laut terbesar.",
            "Jembatan gantung itu menghubungkan dua desa terpencil.",
        ],
    },
    {
        "q": "Apa fungsi akar pada tumbuhan?",
        "correct": "Bagian tanaman yang menancap di dalam tanah berfungsi menyerap air dan unsur hara.",
        "distractors": [
            "Liburan sekolah dimulai pada akhir bulan ini.",
            "Pelukis itu menggelar pameran tunggal di galeri.",
            "Harga emas dunia menyentuh rekor tertinggi.",
            "Stasiun luar angkasa mengorbit Bumi setiap 90 menit.",
        ],
    },
    {
        "q": "Bagaimana cara menghemat penggunaan listrik di rumah?",
        "correct": "Mematikan peralatan elektronik saat tidak digunakan dapat mengurangi konsumsi daya.",
        "distractors": [
            "Klub sepak bola itu menunjuk pelatih baru.",
            "Hutan bakau melindungi pantai dari abrasi.",
            "Novel terbaru penulis itu masuk daftar terlaris.",
            "Pasar tradisional ramai menjelang hari raya.",
        ],
    },
    {
        "q": "Apa penyebab utama pemanasan global?",
        "correct": "Emisi gas rumah kaca dari pembakaran bahan bakar fosil meningkatkan suhu rata-rata bumi.",
        "distractors": [
            "Tarian tradisional itu diiringi alat musik gamelan.",
            "Bandara internasional baru mulai beroperasi.",
            "Anak-anak gemar bermain layang-layang saat sore.",
            "Toko buku itu memberikan diskon besar akhir tahun.",
        ],
    },
    {
        "q": "Siapa proklamator kemerdekaan Indonesia?",
        "correct": "Soekarno dan Mohammad Hatta membacakan teks proklamasi pada 17 Agustus 1945.",
        "distractors": [
            "Kopi robusta banyak ditanam di dataran tinggi.",
            "Aplikasi pembayaran digital semakin populer.",
            "Hujan deras menyebabkan genangan di beberapa ruas jalan.",
            "Spesies orangutan terancam karena hilangnya habitat.",
        ],
    },
]


def local_topk(onnx_dir, q, documents):
    from flashindorank import CustomReranker
    from flashrank import RerankRequest

    if not hasattr(local_topk, "_ranker"):
        local_topk._ranker = CustomReranker(onnx_dir)
    passages = [{"id": i, "text": d} for i, d in enumerate(documents)]
    out = local_topk._ranker.rerank(RerankRequest(query=q, passages=passages))
    return out[0]["id"]


def openrouter_topk(model, q, documents, key):
    body = json.dumps({"model": model, "query": q, "documents": documents}).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/madebyaris/flashIndorank", "X-Title": "flashIndorank"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    best = max(data["results"], key=lambda r: r["relevance_score"])
    return best["index"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default="models/ft-id-ce-onnx")
    parser.add_argument("--openrouter-models", nargs="+",
                        default=["cohere/rerank-v3.5", "nvidia/llama-nemotron-rerank-vl-1b-v2:free"])
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY in the environment.")

    models = ["ours (ft-id-ce ONNX)"] + args.openrouter_models
    correct_counts = {m: 0 for m in models}

    print(f"{'#':>2}  {'question':<48}" + "".join(f"{m.split('/')[-1][:24]:>26}" for m in models))
    print("-" * (2 + 2 + 48 + 26 * len(models)))
    for i, item in enumerate(QUIZ, 1):
        documents = [item["correct"]] + item["distractors"]  # index 0 = correct
        marks = []
        # ours
        ours_idx = local_topk(args.onnx_dir, item["q"], documents)
        ok = ours_idx == 0
        correct_counts[models[0]] += ok
        marks.append("correct" if ok else "WRONG")
        # openrouter
        for m in args.openrouter_models:
            try:
                idx = openrouter_topk(m, item["q"], documents, key)
                ok = idx == 0
            except Exception as e:  # noqa: BLE001
                print(f"   ! {m}: {repr(e)[:120]}", file=sys.stderr)
                ok = False
            correct_counts[m] += ok
            marks.append("correct" if ok else "WRONG")
            time.sleep(0.3)
        print(f"{i:>2}  {item['q'][:48]:<48}" + "".join(f"{mk:>26}" for mk in marks))

    print("-" * (2 + 2 + 48 + 26 * len(models)))
    n = len(QUIZ)
    print(f"{'':>2}  {'TOTAL correct (top-1)':<48}" + "".join(f"{str(correct_counts[m])+'/'+str(n):>26}" for m in models))


if __name__ == "__main__":
    main()
