from typing import Dict, List, Tuple
import re
from collections import Counter
from utils.logger import setup_logger
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict

logger = setup_logger()


class SEOAnalyzer:
    def __init__(self):
        """Initialize NLTK data and syllable dictionary"""
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/cmudict")
        except LookupError:
            nltk.download("punkt")
            nltk.download("cmudict")

        # Load CMU dictionary for syllable counting
        self.syllable_dict = cmudict.dict()

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using CMU dictionary"""
        word = word.lower()
        try:
            return len([ph for ph in self.syllable_dict[word][0] if ph[-1].isdigit()])
        except KeyError:
            # Fallback: count vowel groups if word not in dictionary
            return len(re.findall("[aeiouy]+", word))

    def calculate_keyword_density(
        self, content: str, keywords: List[str]
    ) -> Dict[str, float]:
        """Calculate keyword density for each keyword"""
        content_lower = content.lower()
        words = word_tokenize(content_lower)
        total_words = len(words)

        density = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Handle multi-word keywords
            if " " in keyword_lower:
                count = content_lower.count(keyword_lower)
            else:
                count = sum(1 for word in words if word == keyword_lower)
            density[keyword] = (count / total_words) * 100

        return density

    def analyze_readability(self, content: str) -> Dict[str, float]:
        """Analyze content readability using NLTK"""
        try:
            # Use NLTK's tokenizers for more accurate analysis
            sentences = sent_tokenize(content)
            words = word_tokenize(content)

            # Filter out punctuation from words
            words = [word for word in words if word.isalnum()]

            # Calculate metrics
            num_sentences = len(sentences)
            num_words = len(words)

            # Calculate syllables for each word
            syllable_counts = [self.count_syllables(word) for word in words]
            total_syllables = sum(syllable_counts)

            # Calculate averages
            avg_sentence_length = num_words / num_sentences if num_sentences else 0
            avg_syllables_per_word = total_syllables / num_words if num_words else 0

            # Calculate Flesch Reading Ease score
            # Formula: 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)
            flesch_score = (
                206.835
                - (1.015 * avg_sentence_length)
                - (84.6 * avg_syllables_per_word)
            )

            return {
                "flesch_reading_ease": max(0, min(100, flesch_score)),
                "avg_sentence_length": avg_sentence_length,
                "avg_syllables_per_word": avg_syllables_per_word,
                "total_words": num_words,
                "total_sentences": num_sentences,
                "complex_words": sum(1 for count in syllable_counts if count >= 3),
                "avg_word_length": sum(len(word) for word in words) / num_words
                if num_words
                else 0,
            }
        except Exception as e:
            logger.error(f"Error in readability analysis: {str(e)}")
            raise

    def check_content_structure(self, content: str) -> Dict[str, bool]:
        """Check content structure for SEO best practices"""
        sentences = sent_tokenize(content)

        # Improved header detection
        h1_pattern = r"^#\s+[^\n]+"
        h2_pattern = r"^##\s+[^\n]+"
        h3_pattern = r"^###\s+[^\n]+"

        return {
            "has_h1": bool(re.search(h1_pattern, content, re.MULTILINE)),
            "has_h2": bool(re.search(h2_pattern, content, re.MULTILINE)),
            "has_h3": bool(re.search(h3_pattern, content, re.MULTILINE)),
            "has_lists": bool(re.search(r"^[-*]\s+[^\n]+", content, re.MULTILINE)),
            "has_paragraphs": len([s for s in sentences if s.strip()]) > 1,
            "avg_paragraph_length": len(sentences) / (content.count("\n\n") + 1),
        }

    def generate_seo_score(self, metrics: Dict) -> Tuple[float, List[str]]:
        """Generate overall SEO score and improvement suggestions"""
        score = 100
        improvements = []

        # Keyword density checks
        for keyword, density in metrics["keyword_density"].items():
            if density < 0.5:
                score -= 5
                improvements.append(
                    f"Increase usage of keyword '{keyword}' (current density: {density:.1f}%)"
                )
            elif density > 2.5:
                score -= 5
                improvements.append(
                    f"Reduce usage of keyword '{keyword}' (current density: {density:.1f}%)"
                )

        # Readability checks
        readability = metrics["readability"]
        if readability["flesch_reading_ease"] < 60:
            score -= 10
            improvements.append(
                f"Improve readability (current score: {readability['flesch_reading_ease']:.1f})"
            )

        if readability["avg_sentence_length"] > 25:
            score -= 5
            improvements.append("Sentences are too long. Try to keep them shorter.")

        # Structure checks
        structure = metrics["structure"]
        if not structure["has_h1"]:
            score -= 5
            improvements.append("Add a main heading (H1)")
        if not structure["has_h2"]:
            score -= 5
            improvements.append("Add subheadings (H2) to break up content")
        if not structure["has_lists"]:
            score -= 5
            improvements.append("Add bullet points or numbered lists")

        if structure.get("avg_paragraph_length", 0) > 5:
            score -= 5
            improvements.append("Break up long paragraphs for better readability")

        return max(0, min(score, 100)), improvements

    def analyze_content(self, content: str, keywords: List[str]) -> Dict:
        """Perform comprehensive SEO analysis of content"""
        try:
            keyword_density = self.calculate_keyword_density(content, keywords)
            readability = self.analyze_readability(content)
            structure = self.check_content_structure(content)

            metrics = {
                "keyword_density": keyword_density,
                "readability": readability,
                "structure": structure,
                "content_length": len(word_tokenize(content)),
            }

            score, improvements = self.generate_seo_score(metrics)

            return {
                "score": score,
                "metrics": metrics,
                "improvements": improvements,
                "recommended_length": "1000-1500 words"
                if metrics["content_length"] < 1000
                else "Good length",
                "primary_keyword": keywords[0] if keywords else None,
                "secondary_keywords": keywords[1:] if len(keywords) > 1 else [],
            }

        except Exception as e:
            logger.error(f"Error in SEO analysis: {str(e)}")
            raise

    def suggest_keywords(self, content: str, main_keywords: List[str]) -> List[str]:
        """Suggest additional keywords based on content analysis"""
        try:
            # Use NLTK's word_tokenize for better accuracy
            words = word_tokenize(content.lower())

            # Remove punctuation and short words
            words = [word for word in words if word.isalnum() and len(word) > 4]

            # Get word frequency with Counter
            word_freq = Counter(words)

            # Filter out main keywords and common words
            suggestions = [
                word
                for word, freq in word_freq.most_common(30)
                if word not in main_keywords
                and freq > 1  # Appear more than once
                and len(word) > 4  # Longer than 4 characters
            ][:10]

            return suggestions
        except Exception as e:
            logger.error(f"Error generating keyword suggestions: {str(e)}")
            raise

