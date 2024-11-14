import streamlit as st
import re
import json
from content.extractor import ContentExtractor
from content.generator import ContentGenerator
from utils.logger import setup_logger
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger()


class BlogGeneratorUI:
    def __init__(self):
        self.content_extractor = ContentExtractor()
        self.content_generator = ContentGenerator()
        self.setup_session_state()
        self.setup_page_config()

    def setup_session_state(self):
        """Initialize session state variables"""
        defaults = {
            "extracted_content": None,
            "keywords": [],
            "generated_content": None,
            "edited_content": "",
            "blog_title": "",  # Add title to session state
        }
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def setup_page_config(self):
        """Configure Streamlit page settings and styling"""
        st.set_page_config(
            page_title="CyberSec Blog Generator",
            page_icon="📝",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        st.markdown(
            """
            <style>
                .main > div {
                    padding-top: 2rem;
                }
                .stTextInput > div > div > input {
                    padding: 0.75rem 1rem;
                    border-radius: 0.375rem;
                }
                .stButton > button,
                button[kind="primary"],
                .stFormSubmitter > button,
                .main .element-container div[data-testid="stFormSubmitButton"] button {
                    background-color: rgb(79, 70, 229) !important;
                    border: none !important;
                    color: white !important;
                    padding: 0.75rem 1rem !important;
                    font-weight: 500 !important;
                    border-radius: 0.375rem !important;
                    width: 100% !important;
                    box-shadow: none !important;
                }
                .stButton > button:hover,
                button[kind="primary"]:hover,
                .stFormSubmitter > button:hover {
                    background-color: rgb(67, 56, 202) !important;
                }
                .preview-box {
                    padding: 1rem;
                    border: 1px solid #E5E7EB;
                    border-radius: 0.5rem;
                    background-color: white;
                    margin-top: 1rem;
                }
                .seo-score {
                    background-color: #FEF9C3;
                    padding: 2rem;
                    border-radius: 0.5rem;
                    text-align: center;
                }
                .success-message {
                    padding: 1rem;
                    border-radius: 0.375rem;
                    background-color: #ECFDF5;
                    color: #065F46;
                    margin: 1rem 0;
                }
                .editor-container {
                    margin-bottom: 2rem;
                }
                .markdown-preview {
                    margin-top: 2rem;
                    padding: 1rem;
                    border: 1px solid #E5E7EB;
                    border-radius: 0.5rem;
                    background-color: white;
                }
                .keyword-suggestion {
                    background-color: #F9FAFB;
                    padding: 0.75rem;
                    border-radius: 0.5rem;
                    margin-bottom: 0.75rem;
                }
                .keyword-volume {
                    color: #6B7280;
                    font-size: 0.875rem;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_header(self):
        """Render the application header"""
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='font-size: 2.5rem; font-weight: bold;'>CyberSec Blog Generator</h1>
                <p style='color: #6B7280; font-size: 1.1rem;'>Create SEO-optimized cybersecurity content with AI assistance</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_keyword_suggestions(self):
        """Render keyword suggestions panel"""
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <h3 style='margin: 0; font-size: 1.1rem;'>Keyword Suggestions</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        suggestions = [
            {
                "keyword": "cybersecurity threats",
                "volume": "25K",
                "difficulty": "Medium",
            },
            {"keyword": "network security", "volume": "18K", "difficulty": "High"},
            {
                "keyword": "security best practices",
                "volume": "15K",
                "difficulty": "Low",
            },
        ]

        for suggestion in suggestions:
            st.markdown(
                f"""
                <div class='keyword-suggestion'>
                    <div style='font-weight: 500;'>{suggestion['keyword']}</div>
                    <div class='keyword-volume'>
                        Volume: {suggestion['volume']} • Difficulty: {suggestion['difficulty']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def render_input_section(self):
        """Render URL and keywords input section"""
        st.markdown("### Input Details")

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.form("url_form"):
                st.markdown(
                    """
                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                        </svg>
                        <span>Reference URL</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                url = st.text_input(
                    "Reference URL",
                    placeholder="https://example.com/article",
                    label_visibility="collapsed",
                )

                st.markdown(
                    """
                    <div style='display: flex; align-items: center; gap: 0.5rem; margin: 1rem 0 0.5rem;'>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path>
                            <line x1="7" y1="7" x2="7.01" y2="7"></line>
                        </svg>
                        <span>Target Keywords</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                keywords = st.text_area(
                    "Target Keywords",
                    placeholder="Enter keywords separated by commas",
                    height=100,
                    label_visibility="collapsed",
                )

                submitted = st.form_submit_button(
                    "Generate Content",
                    use_container_width=True,
                )

        with col2:
            self.render_keyword_suggestions()

        if submitted and url:
            return url, keywords
        return None, None

    def process_markdown(self, content: str) -> str:
        """Process markdown content for preview"""
        if not content:
            return ""

        # Handle Title and Meta Description
        if "Title Tag:" in content:
            content = content.replace("Title Tag:", '<div class="title-tag">Title Tag:')
            content = content.replace(
                "\nMeta Description:",
                '</div>\n<div class="meta-description">Meta Description:',
            )
            content = content.replace("\n\nIn ", "</div>\n\nIn ")

        # Process headers
        content = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", content, flags=re.MULTILINE)
        content = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", content, flags=re.MULTILINE)
        content = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", content, flags=re.MULTILINE)
        content = re.sub(r"^#### (.*?)$", r"<h4>\1</h4>", content, flags=re.MULTILINE)

        # Process lists
        content = re.sub(r"^\- (.*?)$", r"• \1", content, flags=re.MULTILINE)
        content = re.sub(r"^\d\. (.*?)$", r"<li>\1</li>", content, flags=re.MULTILINE)

        # Process emphasis
        content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
        content = re.sub(r"\*(.*?)\*", r"<em>\1</em>", content)

        # Process paragraphs
        paragraphs = content.split("\n\n")
        processed_paragraphs = []
        for p in paragraphs:
            if p.strip():
                if not (
                    p.startswith("<h") or p.startswith("•") or p.startswith("<div")
                ):
                    p = f"<p>{p}</p>"
                processed_paragraphs.append(p)

        return "\n".join(processed_paragraphs)

    def clean_and_parse_json_response(self, response: str) -> dict:
        """Parse JSON response from generator"""
        try:
            json_match = re.search(r"{.*}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                content_dict = json.loads(json_str)
                return {
                    "title": content_dict.get("title", ""),
                    "blog_post": content_dict.get("blog_post", ""),
                    "seo_score": content_dict.get("seo_score", "0/100"),
                    "optimization_tips": content_dict.get("optimization_tips", []),
                }
            else:
                logger.warning("No JSON content found in response")
                return {
                    "title": "",
                    "blog_post": response,
                    "seo_score": "0/100",
                    "optimization_tips": [],
                }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {
                "title": "",
                "blog_post": response,
                "seo_score": "0/100",
                "optimization_tips": [],
            }

    def format_optimization_tips(self, tips) -> list:
        """Format optimization tips for display"""
        formatted_tips = []
        if isinstance(tips, dict):
            for category, items in tips.items():
                formatted_tips.extend([f"• {item}" for item in items])
        elif isinstance(tips, list):
            formatted_tips.extend([f"• {tip}" for tip in tips])
        elif isinstance(tips, str):
            formatted_tips.append(f"• {tips}")
        return formatted_tips

    def run(self):
        """Run the Streamlit application"""
        self.render_header()
        url, keywords = self.render_input_section()

        if url:
            try:
                with st.spinner("Processing content..."):
                    content = self.content_extractor.extract_content(url)
                    st.session_state.extracted_content = content
                    st.session_state.keywords = [
                        k.strip() for k in keywords.split(",") if k.strip()
                    ]

                    generated_result = self.content_generator.generate_complete_post(
                        content,
                        st.session_state.keywords,
                    )

                    cleaned_result = (
                        self.content_generator.clean_and_parse_json_response(
                            generated_result.get("raw_response", "")
                        )
                    )

                    # Update session state with all content including title
                    st.session_state.generated_content = cleaned_result
                    st.session_state.edited_content = cleaned_result.get(
                        "blog_post", ""
                    )
                    st.session_state.blog_title = cleaned_result.get("title", "")

                st.markdown(
                    """
                    <div class='success-message'>
                        Content generated successfully! Review and edit below.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

        self.render_content_editor_and_preview()

    def render_content_editor_and_preview(self):
        """Render content editor and preview sections"""
        if st.session_state.generated_content:
            st.markdown("### Content Editor")

            left_col, right_col = st.columns([7, 3])

            with left_col:
                # Title input field
                title = st.text_input(
                    "Title",
                    value=st.session_state.blog_title,
                    key="title_editor",
                    placeholder="Enter blog post title",
                    help="Edit the blog post title",
                )
                st.session_state.blog_title = title

                # Content editor
                edited_content = st.text_area(
                    "Edit Generated Content",
                    value=st.session_state.edited_content,
                    height=400,
                    key="content_editor",
                )
                st.session_state.edited_content = edited_content

                if edited_content:
                    st.markdown(
                        """
                        <div class="markdown-preview">
                        <h4>Preview</h4>
                        """,
                        unsafe_allow_html=True,
                    )
                    if title:
                        st.markdown(f"# {title}")
                    preview_content = self.process_markdown(edited_content)
                    st.markdown(preview_content, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                # Display SEO Score
                if st.session_state.generated_content.get("seo_score"):
                    score = st.session_state.generated_content["seo_score"]
                    score_match = re.search(r"(\d+)", score)
                    score_value = score_match.group(1) if score_match else "0"

                    st.markdown(
                        f"""
                        <div class='seo-score'>
                            <h1 style='font-size: 3rem; font-weight: bold; color: #854D0E; margin: 0;'>{score_value}</h1>
                            <div style='color: #6B7280;'>out of 100</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Display Optimization Tips
                st.markdown("### Optimization Tips")
                tips = st.session_state.generated_content.get("optimization_tips", [])

                # Debug logging
                logger.info(f"Tips before rendering: {json.dumps(tips, indent=2)}")

                if isinstance(tips, str):
                    try:
                        tips = json.loads(tips)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse tips JSON: {tips}")

                if tips:
                    self.render_optimization_tips(tips)
                else:
                    st.info("No optimization tips available")

    def render_optimization_tips(self, tips):
        """Render optimization tips with improved formatting"""
        try:
            # Debug logging
            logger.info(f"Rendering tips: {json.dumps(tips, indent=2)}")

            for tip_group in tips:
                if isinstance(tip_group, dict):
                    category = tip_group.get("category", "")
                    tip_items = tip_group.get("tips", [])

                    if category:
                        st.markdown(
                            f"""
                            <div style='margin-top: 1rem;'>
                                <strong style='font-size: 1.1rem; color: #1F2937;'>{category}</strong>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    for tip in tip_items:
                        st.markdown(
                            f"""
                            <div style='display: flex; align-items: start; gap: 0.5rem; margin: 0.5rem 0 0.5rem 1rem;'>
                                <div style='width: 8px; height: 8px; margin-top: 0.5rem; border-radius: 50%; background-color: #22C55E; flex-shrink: 0;'></div>
                                <div style='color: #374151;'>{tip}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        except Exception as e:
            logger.error(f"Error rendering optimization tips: {str(e)}")
            st.error("Error displaying optimization tips")


if __name__ == "__main__":
    ui = BlogGeneratorUI()
    ui.run()
