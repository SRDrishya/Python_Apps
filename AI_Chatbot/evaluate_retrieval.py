from rag import RAG


TEST_CASES = [
    {
        "question": "What is the leave policy?",
        "expected_source": "leave_policy.pdf",
        "expected_page": 1,
    },
    {
        "question": "Who can approve leave?",
        "expected_source": "leave_policy.pdf",
        "expected_page": 1,
    },
    {
        "question": "What is the reimbursement period?",
        "expected_source": "reimbursement_policy.pdf",
        "expected_page": 1,
    },
]


def matches_expected(result, test_case):
    metadata = result.get("metadata", {})
    return (
        metadata.get("source") == test_case["expected_source"]
        and metadata.get("page") == test_case["expected_page"]
    )


def evaluate_retrieval():
    # Use five unfiltered neighbors so evaluation measures the retriever,
    # rather than hiding low-scoring results behind the production threshold.
    retriever = RAG(top_k=5, min_score=0)
    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_rank_total = 0.0

    for test_case in TEST_CASES:
        results = retriever.retrieve(test_case["question"])
        hit_at_1 = bool(results) and matches_expected(results[0], test_case)
        hit_at_5 = any(
            matches_expected(result, test_case)
            for result in results[:5]
        )

        hits_at_1 += hit_at_1
        hits_at_5 += hit_at_5
        reciprocal_rank_total += next(
            (
                1 / rank
                for rank, result in enumerate(results, start=1)
                if matches_expected(result, test_case)
            ),
            0.0,
        )

        print(f"\nQuestion: {test_case['question']}")
        print(
            "Expected: "
            f"{test_case['expected_source']}, "
            f"page {test_case['expected_page']}"
        )
        print("Top 5 chunks:")

        for rank, result in enumerate(results[:5], start=1):
            metadata = result.get("metadata", {})
            print(
                f"{rank}. score={result['score']:.4f} | "
                f"source={metadata.get('source', 'unknown')} | "
                f"page={metadata.get('page', 'unknown')}"
            )
            print(f"   {result['text'][:180]}")

        print(f"Correct chunk in top 1: {'yes' if hit_at_1 else 'no'}")
        print(f"Correct chunk in top 5: {'yes' if hit_at_5 else 'no'}")

    total = len(TEST_CASES)
    print("\nMetrics:")
    print(f"Recall@1: {hits_at_1}/{total} = {hits_at_1 / total:.1%}")
    print(f"Recall@5: {hits_at_5}/{total} = {hits_at_5 / total:.1%}")
    print(f"MRR: {reciprocal_rank_total / total:.3f}")


if __name__ == "__main__":
    evaluate_retrieval()
