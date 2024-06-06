import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Total number of pages in the corpus
    num_pages = len(corpus)
    
    # Initialize the probability distribution dictionary
    probabilities = {}

    # List of all pages in the corpus
    all_pages = list(corpus.keys())
    
    # List of linked pages from the current page
    linked_pages = corpus[page]
    
    # If the current page has no links, treat it as linking to all pages
    if len(linked_pages) == 0:
        linked_pages = all_pages

    # Calculate the probabilities
    for p in all_pages:
        probabilities[p] = (1 - damping_factor) / num_pages
        if p in linked_pages:
            probabilities[p] += damping_factor / len(linked_pages)
    
    return probabilities

# Example usage
corpus = {
    "1.html": {"2.html", "3.html"},
    "2.html": {"3.html"},
    "3.html": {"2.html"}
}
page = "1.html"
damping_factor = 0.85

print(transition_model(corpus, page, damping_factor))


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize visit counts for each page
    page_counts = {page: 0 for page in corpus}

    # Start with a random page
    current_page = random.choice(list(corpus.keys()))

    # Perform the random walk for n steps
    for _ in range(n):
        page_counts[current_page] += 1
        # Determine the next page using the transition model
        probabilities = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]
        current_page = next_page

    # Convert counts to probabilities (PageRank values)
    pagerank = {page: count / n for page, count in page_counts.items()}

    return pagerank

# Example usage
corpus = {
    "1.html": {"2.html", "3.html"},
    "2.html": {"3.html"},
    "3.html": {"2.html"}
}
damping_factor = 0.85
n = 10000

print(sample_pagerank(corpus, damping_factor, n))

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    def iterate_pagerank(corpus, damping_factor, convergence_threshold=0.001):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Total number of pages in the corpus
    num_pages = len(corpus)

    # Initialize PageRank values
    pagerank = {page: 1 / num_pages for page in corpus}

    # Iterate until convergence
    converged = False
    while not converged:
        new_pagerank = {}
        for page in corpus:
            rank_sum = 0
            for p in corpus:
                if page in corpus[p]:
                    rank_sum += pagerank[p] / len(corpus[p])
                if len(corpus[p]) == 0:
                    rank_sum += pagerank[p] / num_pages
            new_pagerank[page] = (1 - damping_factor) / num_pages + damping_factor * rank_sum

        # Check for convergence
        converged = True
        for page in pagerank:
            if abs(new_pagerank[page] - pagerank[page]) > convergence_threshold:
                converged = False

        # Update PageRank values
        pagerank = new_pagerank

    return pagerank


if __name__ == "__main__":
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"}       
    }
    damping_factor = 0.85
    pagerank = iterate_pagerank(corpus, damping_factor)
    for page, rank in pagerank.items():
        print(f"{page}: {rank:.4f}")
