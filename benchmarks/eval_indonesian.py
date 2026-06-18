"""Quality eval for Indonesian (Bahasa Indonesia) reranking.

This is a small, transparent relevance eval. Each query has exactly one
*relevant* passage that is a **paraphrase** of the query (synonyms, low lexical
overlap) plus several clearly-irrelevant distractors. A reranker that truly
"understands" Indonesian should rank the paraphrase first even without word
overlap.

Reported metrics:
* top-1 accuracy - fraction of queries where the relevant passage is ranked #1
* MRR - mean reciprocal rank of the relevant passage

Run (venv active):

    python benchmarks/eval_indonesian.py

Add your own (query, relevant, distractors) rows to EVAL_SET to harden it.
"""

from __future__ import annotations

import argparse
import warnings

from flashindorank import rerank

warnings.filterwarnings("ignore")

# id of the relevant passage is stored under "rel".
EVAL_SET = [
    {
        "q": "Bagaimana cara menurunkan berat badan?",
        "passages": {
            10: "Olahraga teratur dan pola makan sehat membantu mengurangi bobot tubuh.",
            11: "Mobil listrik semakin populer di kota-kota besar dunia.",
            12: "Harga emas global naik tajam dalam sepekan terakhir.",
            13: "Resep nasi goreng yang lezat dan mudah dibuat di rumah.",
        },
        "rel": 10,
    },
    {
        "q": "Siapa penemu bola lampu?",
        "passages": {
            20: "Thomas Edison dikenal karena mengembangkan lampu pijar yang praktis.",
            21: "Sepak bola adalah olahraga paling populer di dunia.",
            22: "Gunung Merapi merupakan salah satu gunung berapi paling aktif.",
            23: "Kopi luwak berasal dari biji kopi yang dimakan musang.",
        },
        "rel": 20,
    },
    {
        "q": "Apa manfaat tidur yang cukup bagi tubuh?",
        "passages": {
            30: "Istirahat malam yang berkualitas meningkatkan daya ingat dan suasana hati.",
            31: "Banjir melanda beberapa wilayah ibu kota akibat hujan deras.",
            32: "Saham teknologi mengalami koreksi tajam kemarin sore.",
            33: "Festival budaya tahunan menarik ribuan wisatawan asing.",
        },
        "rel": 30,
    },
    {
        "q": "Kenapa langit tampak berwarna biru?",
        "passages": {
            40: "Hamburan cahaya matahari oleh atmosfer membuat angkasa terlihat kebiruan.",
            41: "Pemerintah meresmikan jalan tol baru di Pulau Sumatra.",
            42: "Tim nasional meraih kemenangan penting di laga tandang.",
            43: "Inflasi tahunan tercatat melandai pada kuartal ini.",
        },
        "rel": 40,
    },
    {
        "q": "Bagaimana proses fotosintesis pada tumbuhan?",
        "passages": {
            50: "Daun mengubah sinar matahari, air, dan karbon dioksida menjadi energi dan oksigen.",
            51: "Bank sentral memutuskan menahan suku bunga acuan bulan ini.",
            52: "Penjualan ponsel pintar meningkat menjelang akhir tahun.",
            53: "Rendang dinobatkan sebagai salah satu makanan terlezat dunia.",
        },
        "rel": 50,
    },
]

DEFAULT_MODELS = [
    "ms-marco-TinyBERT-L-2-v2",
    "ms-marco-MiniLM-L-12-v2",
    "ms-marco-MultiBERT-L-12",
]


def evaluate(model_name: str) -> tuple[float, float, list[int]]:
    hits = 0
    rr = 0.0
    ranks = []
    for item in EVAL_SET:
        passages = [{"id": k, "text": v} for k, v in item["passages"].items()]
        results = rerank(item["q"], passages, model_name=model_name)
        ranked_ids = [r["id"] for r in results]
        rank = ranked_ids.index(item["rel"]) + 1
        ranks.append(rank)
        if rank == 1:
            hits += 1
        rr += 1.0 / rank
    n = len(EVAL_SET)
    return hits / n, rr / n, ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    print("Indonesian reranking eval (paraphrased relevant passage, low lexical overlap)\n")
    print(f"{'model':<28}{'top1_acc':>10}{'MRR':>8}   ranks")
    for model_name in args.models:
        acc, mrr, ranks = evaluate(model_name)
        print(f"{model_name:<28}{acc:>10.2f}{mrr:>8.3f}   {ranks}")


if __name__ == "__main__":
    main()
