"""Final, HARD head-to-head on fresh Indonesian questions.

Unlike quiz_new.py, every distractor is about the SAME topic/entity as the
query but answers a DIFFERENT facet (when vs where vs who vs how-many ...). A
reranker must understand the precise intent of the question, not just the topic.

Auth: set OPENROUTER_API_KEY. Run:
    python training/quiz_hard.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quiz_new import local_topk, openrouter_topk  # noqa: E402

# Each item: query + correct passage + same-topic "near miss" distractors.
HARD_QUIZ = [
    {
        "q": "Di mana letak Candi Borobudur?",
        "correct": "Candi Borobudur terletak di Kabupaten Magelang, Jawa Tengah.",
        "distractors": [
            "Candi Borobudur dibangun pada masa Dinasti Syailendra sekitar abad ke-8.",
            "Candi Borobudur merupakan candi Buddha bertingkat dengan ratusan stupa.",
            "Candi Borobudur pernah dipugar besar-besaran dengan bantuan UNESCO.",
            "Relief di dinding Candi Borobudur menggambarkan ajaran dan kehidupan Buddha.",
        ],
    },
    {
        "q": "Kapan Indonesia memproklamasikan kemerdekaannya?",
        "correct": "Indonesia memproklamasikan kemerdekaannya pada tanggal 17 Agustus 1945.",
        "distractors": [
            "Proklamasi kemerdekaan Indonesia dibacakan oleh Soekarno dan Mohammad Hatta.",
            "Teks proklamasi dirumuskan di rumah Laksamana Maeda.",
            "Naskah proklamasi diketik oleh Sayuti Melik.",
            "Proklamasi menandai berakhirnya masa penjajahan di Indonesia.",
        ],
    },
    {
        "q": "Siapa yang merumuskan teori relativitas?",
        "correct": "Albert Einstein adalah ilmuwan yang merumuskan teori relativitas.",
        "distractors": [
            "Teori relativitas mengubah pemahaman manusia tentang ruang dan waktu.",
            "Teori relativitas terdiri atas relativitas khusus dan relativitas umum.",
            "Teori relativitas khusus pertama kali diterbitkan pada tahun 1905.",
            "Persamaan E=mc² merupakan salah satu hasil dari teori relativitas.",
        ],
    },
    {
        "q": "Apa ibu kota Provinsi Jawa Barat?",
        "correct": "Ibu kota Provinsi Jawa Barat adalah Kota Bandung.",
        "distractors": [
            "Jawa Barat merupakan provinsi dengan jumlah penduduk terbanyak di Indonesia.",
            "Jawa Barat berbatasan dengan DKI Jakarta dan Jawa Tengah.",
            "Bahasa Sunda banyak digunakan oleh masyarakat Jawa Barat.",
            "Jawa Barat memiliki banyak objek wisata pegunungan yang sejuk.",
        ],
    },
    {
        "q": "Berapa jumlah pemain satu tim sepak bola di lapangan?",
        "correct": "Satu tim sepak bola menurunkan sebelas pemain di lapangan.",
        "distractors": [
            "Pertandingan sepak bola berlangsung dua babak masing-masing empat puluh lima menit.",
            "Penjaga gawang adalah satu-satunya pemain yang boleh menyentuh bola dengan tangan.",
            "Sepak bola dimainkan menggunakan sebuah bola berbentuk bulat.",
            "Seorang wasit memimpin jalannya pertandingan sepak bola.",
        ],
    },
    {
        "q": "Apa fungsi utama paru-paru pada tubuh manusia?",
        "correct": "Paru-paru berfungsi sebagai tempat pertukaran oksigen dan karbon dioksida.",
        "distractors": [
            "Paru-paru manusia berjumlah dua buah dan terletak di dalam rongga dada.",
            "Paru-paru kanan memiliki tiga lobus sedangkan paru-paru kiri dua lobus.",
            "Kebiasaan merokok dapat merusak jaringan paru-paru.",
            "Paru-paru dilindungi oleh tulang-tulang rusuk di sekelilingnya.",
        ],
    },
    {
        "q": "Pada tahun berapa Perang Dunia II berakhir?",
        "correct": "Perang Dunia II berakhir pada tahun 1945.",
        "distractors": [
            "Perang Dunia II dimulai pada tahun 1939.",
            "Perang Dunia II melibatkan negara-negara Sekutu dan Poros.",
            "Perang Dunia II menimbulkan jutaan korban jiwa di seluruh dunia.",
            "Bom atom dijatuhkan di Hiroshima dan Nagasaki menjelang akhir perang.",
        ],
    },
    {
        "q": "Siapa pelukis lukisan Mona Lisa?",
        "correct": "Leonardo da Vinci adalah pelukis yang menciptakan lukisan Mona Lisa.",
        "distractors": [
            "Lukisan Mona Lisa kini dipajang di Museum Louvre, Paris.",
            "Mona Lisa terkenal karena senyumnya yang misterius.",
            "Lukisan Mona Lisa dibuat pada awal abad ke-16.",
            "Mona Lisa termasuk salah satu lukisan paling terkenal di dunia.",
        ],
    },
    {
        "q": "Apa nama mata uang resmi negara Jepang?",
        "correct": "Mata uang resmi negara Jepang adalah yen.",
        "distractors": [
            "Tokyo merupakan ibu kota negara Jepang.",
            "Gunung Fuji adalah gunung tertinggi di Jepang.",
            "Bahasa Jepang menggunakan aksara kanji, hiragana, dan katakana.",
            "Jepang dikenal sebagai negara dengan teknologi yang maju.",
        ],
    },
    {
        "q": "Berapa lama waktu yang dibutuhkan Bumi untuk mengelilingi Matahari?",
        "correct": "Bumi membutuhkan waktu sekitar 365 hari untuk mengelilingi Matahari.",
        "distractors": [
            "Bumi berputar pada porosnya selama sekitar dua puluh empat jam.",
            "Bumi merupakan planet ketiga dari Matahari.",
            "Bumi memiliki satu satelit alami yang bernama Bulan.",
            "Sebagian besar permukaan Bumi tertutup oleh air.",
        ],
    },
]


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
    counts = {m: 0 for m in models}

    print("HARD test: distractors are same-topic near misses (different facet)\n")
    print(f"{'#':>2}  {'question':<58}" + "".join(f"{m.split('/')[-1][:22]:>24}" for m in models))
    print("-" * (4 + 58 + 24 * len(models)))
    for i, item in enumerate(HARD_QUIZ, 1):
        documents = [item["correct"]] + item["distractors"]  # index 0 = correct
        marks = []
        ok = local_topk(args.onnx_dir, item["q"], documents) == 0
        counts[models[0]] += ok
        marks.append("correct" if ok else "WRONG")
        for m in args.openrouter_models:
            try:
                ok = openrouter_topk(m, item["q"], documents, key) == 0
            except Exception as e:  # noqa: BLE001
                print(f"   ! {m}: {repr(e)[:120]}", file=sys.stderr)
                ok = False
            counts[m] += ok
            marks.append("correct" if ok else "WRONG")
            time.sleep(0.3)
        print(f"{i:>2}  {item['q'][:58]:<58}" + "".join(f"{mk:>24}" for mk in marks))

    print("-" * (4 + 58 + 24 * len(models)))
    n = len(HARD_QUIZ)
    print(f"{'':>2}  {'TOTAL correct (top-1)':<58}" + "".join(f"{str(counts[m])+'/'+str(n):>24}" for m in models))


if __name__ == "__main__":
    main()
