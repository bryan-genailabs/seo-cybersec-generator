import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from urllib.parse import urlparse
import logging
from utils.logger import setup_logger

logger = setup_logger()


class ContentExtractor:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            logger.error(f"URL validation error: {str(e)}")
            return False

    def fetch_content(self, url: str) -> str:
        """Fetch content from URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching content: {str(e)}")
            raise Exception(f"Failed to fetch content from {url}: {str(e)}")

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title = soup.find("title")
        if title:
            return title.get_text().strip()

        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()

        return "Untitled"

    def extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description"""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            return meta_desc.get("content", "").strip()
        return None

    def extract_headers(self, soup: BeautifulSoup) -> List[str]:
        """Extract all headers (h1-h6) from content"""
        headers = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            header_text = tag.get_text().strip()
            if header_text:
                headers.append(f"{tag.name}: {header_text}")
        return headers

    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page"""
        # Remove unwanted elements
        for element in soup.find_all(
            ["script", "style", "nav", "header", "footer", "aside"]
        ):
            element.decompose()

        # Try to find main content container
        main_content = None
        possible_content_ids = ["content", "main", "article", "post"]
        possible_content_classes = ["content", "main", "article", "post", "entry"]

        # Try finding by ID
        for content_id in possible_content_ids:
            main_content = soup.find(id=content_id)
            if main_content:
                break

        # Try finding by class
        if not main_content:
            for content_class in possible_content_classes:
                main_content = soup.find(class_=content_class)
                if main_content:
                    break

        # If still not found, use article tag or body
        if not main_content:
            main_content = soup.find("article") or soup.find("body")

        if main_content:
            # Clean up the content
            for tag in main_content.find_all(
                ["script", "style", "nav", "header", "footer", "aside"]
            ):
                tag.decompose()

            # Get text with proper spacing
            content = " ".join(main_content.stripped_strings)
            return content

        return "No content found"

    def extract_content(self, url: str) -> Dict[str, str]:
        """Main method to extract content from URL"""
        if not self.validate_url(url):
            raise ValueError("Invalid URL format")

        html_content = self.fetch_content(url)
        soup = BeautifulSoup(html_content, "html.parser")

        return {
            "title": self.extract_title(soup),
            "meta_description": self.extract_meta_description(soup),
            "headers": self.extract_headers(soup),
            "main_content": self.extract_main_content(soup),
            "url": url,
        }
