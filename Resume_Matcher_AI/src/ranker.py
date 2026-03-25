def rank_candidates(similarity_scores, top_n=3):
    """
    Rank candidates based on similarity scores
    """

    if len(similarity_scores) == 0:
        return []

    ranked = sorted(
        similarity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]


def print_rankings(ranked_list):
    """
    Display rankings in console
    """

    if len(ranked_list) == 0:
        print("No candidates to display.")
        return

    print("\n🏆 Top Candidates:\n")

    for i, (name, score) in enumerate(ranked_list, start=1):
        print(f"{i}. {name} → Score: {score:.4f}")