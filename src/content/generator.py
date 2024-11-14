from openai import OpenAI
from typing import Dict, List, Any, Optional, Union
import os
from utils.logger import setup_logger
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
import json
import re

load_dotenv()
logger = setup_logger()


class ContentGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Initialize RAG components
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Enhanced system prompt with readability and SEO guidelines
        self.system_prompt = """
        You're a helpful A1 assistant that imitates API endpoints for web servers that returns a blog the blog can be about ANY topic. follow these guidelines: You need to imitate this API endpoint in full, replying according to this JSON format: {
        "title": "the title of the blog post"
        "blog_post": "the complete blog post that follows the formatting",
        "seo_score": "seo score/100",
        "optimization_tips": "an array of tips"
        }
        and strictly follow these:
        You are an advanced content generator specialized in creating
highly readable, SEO-optimized blog posts for the cybersecurity niche. Your primary focus
is maintaining a Flesch-Kincaid score above 60 while delivering valuable evergreen content and 
achieving high SEO scores. 

1. **Readability Requirements** (Priority for Flesch-Kincaid Score > 60):
- Use mostly 1-2 syllable words (examples: use "help" not "facilitate", "show" not "demonstrate")
- Keep sentences between 10-15 words on average
- Break any sentence longer than 20 words into two shorter ones
- Use active voice (example: "Hackers breached the system" not "The system was breached")
- Start sentences with familiar words like "This", "Here", "You", "We"
- Use present tense when possible
- Write at an 8th-grade reading level at a maximum
- Voice: Active
- Language: Present tense, common openers(This, Here, You, We)

2. **SEO Optimization Requirements**:
- Keyword Placement:
  * Include primary keyword in first paragraph
  * Use keywords in H1, H2, and H3 headings
  * Place keywords naturally in meta description
  * Include keywords in image alt text
- Keyword Density:
  * Primary keyword: 1.5-2% density
  * Secondary keywords: 0.5-1% density
  * LSI keywords: Spread throughout content
- Meta Elements:
  * Title tag: 50-60 characters with primary keyword
  * Meta description: 150-155 characters with call to action
  * Header tags: H1 > H2 > H3 hierarchy
- Content Structure:
  * 1500-2000 words for pillar content
  * Short paragraphs (2-3 sentences)
  * Bullet points and numbered lists
  * Table of contents for long posts
  * Internal and external links every 300 words

3. **Technical Content Guidelines**:
- Create evergreen content that provides valuable, long-lasting insights and is SEO-friendly
- Define technical terms immediately after use
- Break complex concepts into steps
- Use analogies to explain technical concepts
- Include examples after technical explanations
- Add "In other words..." followed by simpler restatements
- Create comparison tables for technical features

4. **Response Format**:
Your response must be in the following format and should be inside the json structure:
    {
    "title": "the title of the blog post",
    "blog_post": "the complete blog post that follows the formatting",
    "seo_score": "seo score/100",
    "optimization_tips": [
        {
            "category": "Keyword Optimization",
            "tips": [
                "Specific tip 1",
                "Specific tip 2",
                "Specific tip 3"
            ]
        },
        {
            "category": "Content Structure",
            "tips": [
                "Specific tip 1",
                "Specific tip 2",
                "Specific tip 3"
            ]
        },
        {
            "category": "Readability Improvements",
            "tips": [
                "Specific tip 1",
                "Specific tip 2",
                "Specific tip 3"
            ]
        },
        {
            "category": "Technical SEO",
            "tips": [
                "Specific tip 1",
                "Specific tip 2",
                "Specific tip 3"
            ]
        },
        {
            "category": "User Experience",
            "tips": [
                "Specific tip 1",
                "Specific tip 2",
                "Specific tip 3"
            ]
        }
    ]
    }

"""

        # Enhanced user prompt template
        self.user_prompt_template = """now you got an incoming request GET /blog?query=rewrite%20this%20blog%29=
Retrieved Context: {retrieved_context}

Requirements:
1. Start with a news-style paragraph (max 3 sentences, each under 15 words)
2. Use these keywords naturally: {keywords}
3. Break down technical concepts into simple terms
4. Include "Quick Tips" boxes after complex sections
5. Add examples after technical explanations

Content Structure:
- Title: {title}
- Meta Description: {meta_description}
- Preview: {content_preview}

Writing Rules for High Readability:
1. Keep sentences under 15 words
2. Use mostly 1-2 syllable words
3. Start sentences with: This, Here, You, We
4. Break down complex ideas into bullet points
5. Define technical terms right after using them
6. Use active voice
7. Write at 8th-grade level

SEO Requirements:
1. Include primary keyword in first 100 words
2. Add H2 headings every 300 words
3. Include internal/external links
4. Maintain keyword density 1.5-2%
5. Add schema markup suggestions
6. Include social share elements
"""

    def create_knowledge_base(self, content: Dict[str, str]) -> FAISS:
        """Create a structured vector store from the extracted content"""
        try:
            # Split content into different types for better retrieval
            document_sections = [
                # Core content section
                "MAIN CONTENT:\n{}".format(content.get("main_content", "")),
                # Metadata section
                "METADATA:\nTitle: {}\nMeta Description: {}".format(
                    content.get("title", ""), content.get("meta_description", "")
                ),
                # Structure section
                "DOCUMENT STRUCTURE:\n{}".format("\n".join(content.get("headers", []))),
            ]

            # Split each section separately for better chunk coherence
            all_chunks = []
            for section in document_sections:
                chunks = self.text_splitter.split_text(section)
                all_chunks.extend(chunks)

            # Create metadata for each chunk to track its source
            texts_with_sources = []
            for chunk in all_chunks:
                if "MAIN CONTENT:" in chunk:
                    source_type = "main_content"
                elif "METADATA:" in chunk:
                    source_type = "metadata"
                else:
                    source_type = "structure"

                texts_with_sources.append(
                    {
                        "text": chunk,
                        "source": source_type,
                    }
                )

            # Create vector store with metadata
            vectorstore = FAISS.from_texts(
                texts=[t["text"] for t in texts_with_sources],
                embedding=self.embeddings,
                metadatas=texts_with_sources,
            )

            return vectorstore

        except Exception as e:
            logger.error(f"Error creating knowledge base: {str(e)}")
            raise

    def get_relevant_context(self, vectorstore: FAISS, keywords: List[str]) -> str:
        """Retrieve relevant context with improved source awareness"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model_name="gpt-4-turbo-preview", openai_api_key=self.api_key
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 3,
                        "filter": {
                            "source": "main_content"
                        },  # Focus on main content first
                    },
                ),
            )

            contexts = []

            # First get main content context
            for keyword in keywords:
                queries = [
                    f"What are the key technical details about {keyword}?",
                    f"What are the best practices related to {keyword}?",
                    f"What are common challenges or issues with {keyword}?",
                ]

                for query in queries:
                    main_context = qa_chain.run(query)
                    if main_context:
                        contexts.append(
                            f"Technical Details ({keyword}):\n{main_context}"
                        )

            # Then get structural context
            structure_qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model_name="gpt-4-turbo-preview", openai_api_key=self.api_key
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2, "filter": {"source": "structure"}},
                ),
            )

            structure_context = structure_qa_chain.run(
                "What is the main structure and organization of this content?"
            )
            if structure_context:
                contexts.append(f"Content Structure:\n{structure_context}")

            # Finally get metadata context
            metadata_qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model_name="gpt-4-turbo-preview", openai_api_key=self.api_key
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 1, "filter": {"source": "metadata"}},
                ),
            )

            metadata_context = metadata_qa_chain.run(
                "What are the key points from the title and meta description?"
            )
            if metadata_context:
                contexts.append(f"Key Metadata:\n{metadata_context}")

            return "\n\n".join(contexts)

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise

    def generate_complete_post(
        self, content: Dict[str, str], keywords: List[str]
    ) -> Dict[str, str]:
        """Generate a complete blog post with improved logging"""
        try:
            # Log input parameters
            logger.info("Generating content with keywords: %s", keywords)
            logger.info("Content title: %s", content.get("title", "No title"))

            # Create knowledge base and get context
            vectorstore = self.create_knowledge_base(content)
            retrieved_context = self.get_relevant_context(vectorstore, keywords)

            # Log the retrieved context
            logger.info("Retrieved context:")
            logger.info(retrieved_context)

            # Format the prompt and log it
            title = content.get("title", "Untitled")
            meta_description = content.get("meta_description", "")
            content_preview = (
                content.get("main_content", "")[:1000]
                if content.get("main_content")
                else ""
            )

            user_prompt = self.user_prompt_template.format(
                retrieved_context=retrieved_context,
                keywords=", ".join(keywords),
                title=title,
                meta_description=meta_description,
                content_preview=content_preview,
            )

            logger.info("Formatted user prompt:")
            logger.info(user_prompt)

            # Generate content
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=3500,
                presence_penalty=0.6,
                frequency_penalty=0.4,
            )

            # Log the raw response
            logger.info("Raw OpenAI response:")
            logger.info(json.dumps(response.model_dump(), indent=2))

            # Validate and get content
            if not response or not response.choices:
                raise ValueError("Invalid or empty response from OpenAI API")

            generated_content = response.choices[0].message.content.strip()

            # Log the generated content
            logger.info("Generated content before section extraction:")
            logger.info("=" * 80)
            logger.info(generated_content)
            logger.info("=" * 80)

            # Extract sections with logging
            blog_post = self._extract_section(generated_content, "BLOG POST")
            logger.info(
                "Extracted blog post section length: %d",
                len(blog_post) if blog_post else 0,
            )

            seo_score = self._extract_section(generated_content, "SEO SCORE")
            logger.info(
                "Extracted SEO score section length: %d",
                len(seo_score) if seo_score else 0,
            )

            optimization_tips = self._extract_section(
                generated_content, "OPTIMIZATION TIPS"
            )
            logger.info(
                "Extracted optimization tips section length: %d",
                len(optimization_tips) if optimization_tips else 0,
            )

            # Log the final structured response
            result = {
                "title": self._extract_section(generated_content, "TITLE"),
                "blog_post": self._extract_section(generated_content, "BLOG POST"),
                "seo_score": self._extract_section(generated_content, "SEO SCORE"),
                "optimization_tips": self.format_optimization_tips(
                    self._extract_section(generated_content, "OPTIMIZATION TIPS")
                ),
                "raw_response": generated_content,
            }

            logger.info("Processed response:")
            logger.info("Title: %s", result["title"])
            logger.info("Blog post length: %d", len(result["blog_post"]))
            logger.info("SEO score length: %d", len(result["seo_score"]))
            logger.info(
                "Optimization tips length: %d", len(str(result["optimization_tips"]))
            )

            return result

        except Exception as e:
            logger.error("Error generating content: %s", str(e), exc_info=True)
            raise

    def _pre_process_json_content(self, content: str) -> str:
        """Pre-process the JSON content to handle special characters and formatting"""
        # Remove code block markers if present
        content = re.sub(r"^```json\s*|\s*```$", "", content.strip())

        # Extract the JSON object
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return content

        json_str = json_match.group(0)

        # Fix newlines and special characters in blog_post
        def replace_newlines(match):
            text = match.group(1)
            # Escape newlines and special characters
            escaped = text.replace("\n", "\\n").replace("\r", "\\r")
            # Escape quotes
            escaped = escaped.replace('"', '\\"')
            return f'"blog_post": "{escaped}"'

        # Replace blog_post content with escaped version
        json_str = re.sub(
            r'"blog_post":\s*"([\s\S]*?)"(?=,|\s*})', replace_newlines, json_str
        )

        return json_str

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract content from JSON response with improved handling"""
        try:
            if not content:
                return ""

            # Pre-process the content
            processed_content = self._pre_process_json_content(content)

            try:
                # Parse the processed JSON
                content_dict = json.loads(processed_content)

                # Map section names to JSON keys
                section_map = {
                    "TITLE": "title",
                    "BLOG POST": "blog_post",
                    "SEO SCORE": "seo_score",
                    "OPTIMIZATION TIPS": "optimization_tips",
                }

                json_key = section_map.get(section_name)
                if json_key and json_key in content_dict:
                    extracted_content = content_dict[json_key]

                    # Handle different content types
                    if isinstance(extracted_content, (dict, list)):
                        return json.dumps(extracted_content, indent=2)

                    # Unescape newlines in blog post content
                    if json_key == "blog_post":
                        return (
                            extracted_content.replace("\\n", "\n")
                            .replace("\\r", "\r")
                            .replace('\\"', '"')
                        )

                    return str(extracted_content)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {str(e)}")
                return self._extract_with_regex(content, section_name)

            # Return empty string for optional sections
            if section_name in ["SEO SCORE", "OPTIMIZATION TIPS", "TITLE"]:
                return ""

            # For blog post content, return the entire content if markers aren't found
            if section_name == "BLOG POST":
                return content.strip()

            return ""

        except Exception as e:
            logger.error(f"Error extracting {section_name}: {str(e)}")
            return ""

    def _extract_section_legacy(self, content: str, section_name: str) -> str:
        """Legacy method to extract content between section markers"""
        try:
            start_marker = f"[{section_name}]"
            end_marker = f"[/{section_name}]"

            if start_marker in content and end_marker in content:
                start_idx = content.find(start_marker) + len(start_marker)
                end_idx = content.find(end_marker)

                if start_idx >= 0 and end_idx >= 0:
                    return content[start_idx:end_idx].strip()

            return ""

        except Exception as e:
            logger.error(
                "Error in legacy extraction for %s: %s",
                section_name,
                str(e),
                exc_info=True,
            )
            return ""

    def format_optimization_tips(self, tips: str) -> List[Dict[str, Any]]:
        """Format optimization tips for consistent structure"""
        try:
            if not tips:
                return []

            # If tips is a string containing JSON, parse it
            if isinstance(tips, str):
                try:
                    tips = json.loads(tips)
                except json.JSONDecodeError:
                    logger.error("Failed to parse tips JSON")
                    return []

            # If tips is already a list of dictionaries, return as is
            if isinstance(tips, list) and all(isinstance(tip, dict) for tip in tips):
                return tips

            # Convert old format to new format if needed
            if isinstance(tips, list):
                formatted_tips = []
                current_category = None
                current_tips = []

                for item in tips:
                    if isinstance(item, list):
                        # First item in list is category
                        formatted_tips.append({"category": item[0], "tips": item[1:]})
                    elif isinstance(item, str):
                        if ":" in item:
                            # If we have a current category, add it to formatted tips
                            if current_category and current_tips:
                                formatted_tips.append(
                                    {"category": current_category, "tips": current_tips}
                                )
                            # Start new category
                            current_category = item.rstrip(":")
                            current_tips = []
                        else:
                            # Add tip to current category
                            current_tips.append(item.strip("- "))

                # Add last category if exists
                if current_category and current_tips:
                    formatted_tips.append(
                        {"category": current_category, "tips": current_tips}
                    )

                return formatted_tips

            return []

        except Exception as e:
            logger.error(f"Error formatting optimization tips: {str(e)}")
            return []

    def clean_and_parse_json_response(
        self, response: str
    ) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
        """
        Clean and parse JSON response with improved tips handling
        """
        try:
            # Remove code block markers
            response = re.sub(r"^```json\s*|\s*```$", "", response.strip())

            # Find JSON content
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                logger.warning("No JSON object found in response")
                return {
                    "title": "",
                    "blog_post": response,
                    "seo_score": "0/100",
                    "optimization_tips": [],
                }

            # Extract and clean the JSON object
            json_str = json_match.group(0)
            try:
                # Parse the raw JSON first
                content_dict = json.loads(json_str)

                # Handle the optimization tips specially
                tips = content_dict.get("optimization_tips", [])
                if isinstance(tips, str):
                    try:
                        # Try to parse tips if they're a string
                        tips = json.loads(tips)
                    except json.JSONDecodeError:
                        tips = []

                # Ensure consistent format for tips
                formatted_tips = []
                if isinstance(tips, list):
                    for tip in tips:
                        if (
                            isinstance(tip, dict)
                            and "category" in tip
                            and "tips" in tip
                        ):
                            formatted_tips.append(tip)
                        elif isinstance(tip, list) and len(tip) > 0:
                            formatted_tips.append(
                                {
                                    "category": tip[0],
                                    "tips": tip[1:] if len(tip) > 1 else [],
                                }
                            )

                return {
                    "title": content_dict.get("title", ""),
                    "blog_post": content_dict.get("blog_post", ""),
                    "seo_score": content_dict.get("seo_score", "0/100"),
                    "optimization_tips": formatted_tips,
                }

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {str(e)}")
                return {
                    "title": "",
                    "blog_post": response,
                    "seo_score": "0/100",
                    "optimization_tips": [],
                }

        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return {
                "title": "",
                "blog_post": response,
                "seo_score": "0/100",
                "optimization_tips": [],
            }

    def _extract_with_regex(self, content: str, section_name: str) -> str:
        """
        Extract specific sections using regex with improved pattern matching
        """
        patterns = {
            "TITLE": r'"title":\s*"([^"]*?)"',
            "BLOG POST": r'"blog_post":\s*"([\s\S]*?)"(?=\s*,|\s*})',
            "SEO SCORE": r'"seo_score":\s*"([^"]*?)"',
            "OPTIMIZATION TIPS": r'"optimization_tips":\s*(\[[\s\S]*?\])(?=\s*})',
            "FULL": r"\{[\s\S]*\}",
        }

        try:
            if section_name not in patterns:
                return ""

            pattern = patterns[section_name]
            match = re.search(pattern, content, re.DOTALL)

            if not match:
                return ""

            extracted = match.group(1)

            # Clean up the extracted content based on section type
            if section_name == "BLOG POST":
                # Handle newlines and control characters in blog post
                return (
                    extracted.replace("\\n", "\n")
                    .replace("\\r", "\r")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )

            return extracted.strip()

        except Exception as e:
            logger.error(f"Error extracting {section_name}: {str(e)}")
            return ""

    def _parse_tips(self, tips_str: str) -> List[List[str]]:
        """
        Parse optimization tips with improved error handling
        """
        try:
            if not tips_str:
                return []

            # Clean up the tips string
            tips_str = tips_str.strip()

            # Try to parse as JSON
            try:
                tips = json.loads(tips_str)
                if isinstance(tips, list):
                    return tips
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract tips using regex
                tip_pattern = r'\["([^"]+)",\s*"([^"]+)"(?:\s*,\s*"([^"]+)")?\]'
                matches = re.finditer(tip_pattern, tips_str)

                parsed_tips = []
                for match in matches:
                    tip_group = [item for item in match.groups() if item is not None]
                    if tip_group:
                        parsed_tips.append(tip_group)

                return parsed_tips

        except Exception as e:
            logger.error(f"Error parsing optimization tips: {str(e)}")
            return []
