from openai import OpenAI
from typing import Dict, List
import os
from utils.logger import setup_logger
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
import json

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
        You're a helpful A1 assistant that imitates API endpoints for web servers that returns a blog with the guidelines the blog can be about ANY topic. follow these guidelines: You need to imitate this API endpoint in full, replying according to this JSON format: {
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
[TITLE]
{The title of the blog post}
[/TITLE]

[BLOG POST]
{The complete blog post content with proper formatting}
[/BLOG POST]

[SEO SCORE]
Total Score: {score}/100
[/SEO SCORE]

[OPTIMIZATION TIPS]
Provide specific, actionable tips in these categories:
1. Keyword Optimization:
   {bullet points with specific suggestions}
2. Content Structure:
   {bullet points with specific suggestions}
3. Readability Improvements:
   {bullet points with specific suggestions}
4. Technical SEO:
   {bullet points with specific suggestions}
5. User Experience:
   {bullet points with specific suggestions}
[/OPTIMIZATION TIPS]"""

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

Remember to format your response with [BLOG POST], [SEO SCORE], and [OPTIMIZATION TIPS] 
sections as specified in the system prompt."""

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
                temperature=0.7,
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

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract content from JSON response with improved logging"""
        try:
            if not content:
                logger.warning("Empty content provided for %s extraction", section_name)
                return ""

            logger.info("Attempting to extract %s section", section_name)
            logger.info("Content length: %d", len(content))

            # Try to parse JSON from the content
            try:
                # Find the JSON string within the content
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    response_dict = json.loads(json_str)

                    # Map section names to JSON keys
                    section_map = {
                        "TITLE": "title",
                        "BLOG POST": "blog_post",
                        "SEO SCORE": "seo_score",
                        "OPTIMIZATION TIPS": "optimization_tips",
                    }

                    json_key = section_map.get(section_name)
                    if json_key and json_key in response_dict:
                        extracted_content = response_dict[json_key]

                        # Handle different content types
                        if isinstance(extracted_content, (dict, list)):
                            return json.dumps(extracted_content, indent=2)
                        return str(extracted_content)

                    logger.warning(
                        "Section %s not found in JSON response", section_name
                    )

            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON response: %s", str(e))
                # Fall back to original section extraction if JSON parsing fails
                return self._extract_section_legacy(content, section_name)

            # Return empty string for optional sections
            if section_name in ["SEO SCORE", "OPTIMIZATION TIPS", "TITLE"]:
                return ""

            # For blog post content, return the entire content if markers aren't found
            if section_name == "BLOG POST":
                logger.info("Falling back to entire content for blog post")
                return content.strip()

            return ""

        except Exception as e:
            logger.error(
                "Error extracting %s section: %s", section_name, str(e), exc_info=True
            )
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

    def format_optimization_tips(self, tips: str) -> str:
        """Format optimization tips for display"""
        try:
            if not tips:
                return ""

            # If tips is a string containing JSON, parse it
            if isinstance(tips, str):
                try:
                    tips = json.loads(tips)
                except json.JSONDecodeError:
                    return tips

            # If tips is a dictionary, format it nicely
            if isinstance(tips, dict):
                formatted_tips = []
                for category, items in tips.items():
                    formatted_tips.append(f"\n{category}:")
                    if isinstance(items, list):
                        formatted_tips.extend([f"- {item}" for item in items])
                    else:
                        formatted_tips.append(f"- {items}")
                return "\n".join(formatted_tips)

            return str(tips)

        except Exception as e:
            logger.error("Error formatting optimization tips: %s", str(e))
            return str(tips)
