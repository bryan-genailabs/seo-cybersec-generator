import streamlit as st
import re
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
        if "extracted_content" not in st.session_state:
            st.session_state.extracted_content = None
        if "keywords" not in st.session_state:
            st.session_state.keywords = []
        if "generated_content" not in st.session_state:
            st.session_state.generated_content = None
        if "edited_content" not in st.session_state:
            st.session_state.edited_content = ""

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
                background-color: rgb(79, 70, 229) !important;  /* indigo-600 */
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
            .stFormSubmitter > button:hover,
            .main .element-container div[data-testid="stFormSubmitButton"] button:hover {
                background-color: rgb(67, 56, 202) !important;  /* indigo-700 */
                border: none !important;
                box-shadow: none !important;
            }

            .stButton > button:focus,
            button[kind="primary"]:focus,
            .stFormSubmitter > button:focus,
            .main .element-container div[data-testid="stFormSubmitButton"] button:focus {
                box-shadow: none !important;
                outline: none !important;
            }

            /* Ensure the form submit button specifically has correct styles */
            div[data-testid="stFormSubmitButton"] {
                width: 100% !important;
            }
            
            div[data-testid="stFormSubmitButton"] > button {
                background-color: rgb(79, 70, 229) !important;
                color: white !important;
                width: 100% !important;
            }

            div[data-testid="stFormSubmitButton"] > button:hover {
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
                .preview-header {
                    font-size: 1.1rem;
                    font-weight: 500;
                    margin-bottom: 1rem;
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
            {
                "keyword": "network security",
                "volume": "18K",
                "difficulty": "High",
            },
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

    def render_content_editor_and_preview(self):
        """Render content editor and preview sections"""
        if st.session_state.generated_content:
            st.markdown("### Content Editor")

            left_col, right_col = st.columns([7, 3])

            with left_col:
                initial_content = (
                    st.session_state.edited_content
                    if st.session_state.edited_content
                    else st.session_state.generated_content.get("main_content", "")
                )

                edited_content = st.text_area(
                    "Edit Generated Content",
                    value=initial_content,
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

                    preview_content = self.process_markdown(edited_content)
                    st.markdown(preview_content, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                # Display SEO Score
                seo_score = st.session_state.generated_content.get("seo_score", "")
                score_match = re.search(r"Total Score:\s*(\d+)", seo_score)
                score = score_match.group(1) if score_match else "0"

                st.markdown(
                    f"""
                    <div class='seo-score'>
                        <h1 style='font-size: 3rem; font-weight: bold; color: #854D0E; margin: 0;'>{score}</h1>
                        <div style='color: #6B7280;'>out of 100</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Display Optimization Tips
                st.markdown("### Optimization Tips")
                optimization_tips = st.session_state.generated_content.get(
                    "optimization_tips", ""
                )
                if optimization_tips:
                    # Process each line of the optimization tips
                    current_category = None
                    for line in optimization_tips.split("\n"):
                        line = line.strip()
                        if not line:
                            continue

                        # Check if this is a category header
                        if re.match(r"^\d+\.\s+\w+", line):
                            current_category = line
                            st.markdown(f"**{current_category}**")
                        # Check if this is a bullet point
                        elif line.startswith("-"):
                            tip = line[
                                1:
                            ].strip()  # Remove the dash and any leading whitespace
                            st.markdown(
                                f"""
                                <div style='display: flex; align-items: start; gap: 0.5rem; margin: 0.5rem 0 0.5rem 1rem;'>
                                    <div style='width: 8px; height: 8px; margin-top: 0.5rem; border-radius: 50%; background-color: #22C55E; flex-shrink: 0;'></div>
                                    <div style='color: #374151;'>{tip}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

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

    def run(self):
        """Run the Streamlit application"""
        self.render_header()
        url, keywords = self.render_input_section()

        if url:
            try:
                with st.spinner("Processing content..."):
                    # Extract content from URL
                    content = self.content_extractor.extract_content(url)
                    st.session_state.extracted_content = content
                    st.session_state.keywords = [
                        k.strip() for k in keywords.split(",") if k.strip()
                    ]

                    # Generate optimized content
                    generated_content = self.content_generator.generate_complete_post(
                        content,
                        st.session_state.keywords,
                        {},  # Empty dict since we're using model's SEO scoring
                    )
                    st.session_state.generated_content = generated_content
                    st.session_state.edited_content = generated_content.get(
                        "main_content", ""
                    )

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


if __name__ == "__main__":
    ui = BlogGeneratorUI()
    ui.run()
