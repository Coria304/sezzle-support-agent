import glob
import os


def load_all_policies() -> str:
    policy_dir = os.path.join(os.path.dirname(__file__), "POLICIES")
    files = sorted(glob.glob(os.path.join(policy_dir, "POLICY_-*.md")))
    sections = []
    for path in files:
        with open(path) as f:
            sections.append(f.read().strip())
    return "\n\n---\n\n".join(sections)
