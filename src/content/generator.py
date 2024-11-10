from openai import OpenAI
from typing import Dict, List
import os
from utils.logger import setup_logger
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
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
        self.system_prompt = """You are an advanced content generator specialized in creating
highly readable, SEO-optimized blog posts for the cybersecurity niche. Your primary focus
is maintaining a Flesch-Kincaid score above 60 while delivering valuable content and 
achieving high SEO scores. Follow these specific guidelines:

1. **Readability Requirements** (Priority for Flesch-Kincaid Score > 60):
- Use mostly 1-2 syllable words (examples: use "help" not "facilitate", "show" not "demonstrate")
- Keep sentences between 10-15 words on average
- Break any sentence longer than 20 words into two shorter ones
- Use active voice (example: "Hackers breached the system" not "The system was breached")
- Start sentences with familiar words like "This", "Here", "You", "We"
- Use present tense when possible
- Write at an 8th-grade reading level

2. **SEO Optimization Requirements**:
- Keyword Placement:
  * Include primary keyword in first paragraph
  * Use keywords in H1, H2, and H3 headings
  * Place keywords naturally in meta description
  * Include keywords in image alt text
  * Add keywords to URL slug
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
- Define technical terms immediately after use
- Break complex concepts into steps
- Use analogies to explain technical concepts
- Include examples after technical explanations
- Add "In other words..." followed by simpler restatements
- Create comparison tables for technical features

4. **Response Format**:
Your response must be in the following format:

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
        self.user_prompt_template = """Create a highly readable, SEO-optimized cybersecurity blog post
(targeting Flesch-Kincaid score > 60) based on:

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
                        "url": content.get("url", ""),
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
        self, content: Dict[str, str], keywords: List[str], seo_analysis: Dict
    ) -> Dict[str, str]:
        """Generate a complete blog post with RAG-enhanced content, improved readability, and SEO"""
        try:
            # Create knowledge base and get context
            vectorstore = self.create_knowledge_base(content)
            retrieved_context = self.get_relevant_context(vectorstore, keywords)

            # Log retrieved context
            logger.info("Retrieved Context:")
            logger.info(retrieved_context)

            # Use the existing template with context
            user_prompt = self.user_prompt_template.format(
                url=content.get("url", ""),
                retrieved_context=retrieved_context,
                keywords=", ".join(keywords),
                title=content.get("title", ""),
                meta_description=content.get("meta_description", ""),
                content_preview=content.get("main_content", "")[:1000],
            )

            # Generate content using the established system prompt
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

            generated_content = response.choices[0].message.content.strip()

            # Log the complete raw response
            logger.info("Raw response from OpenAI API:")
            logger.info("=" * 80)
            logger.info(generated_content)
            logger.info("=" * 80)

            def extract_section(content: str, section_name: str) -> str:
                logger.info(f"Attempting to extract {section_name} section")
                pattern = (
                    r"\["
                    + re.escape(section_name)
                    + r"\](.*?)\[/"
                    + re.escape(section_name)
                    + r"\]"
                )
                match = re.search(pattern, content, re.DOTALL)

                if not match:
                    logger.warning(f"Failed to find {section_name} section")
                    logger.info(f"Content being searched:\n{content}")
                    return ""

                extracted_content = match.group(1).strip()
                logger.info(f"Successfully extracted {section_name} section:")
                logger.info("-" * 40)
                logger.info(extracted_content)
                logger.info("-" * 40)

                return extracted_content

            # Extract sections using the correct markers
            blog_post = extract_section(generated_content, "BLOG POST")
            seo_score = extract_section(generated_content, "SEO SCORE")
            optimization_tips = extract_section(generated_content, "OPTIMIZATION TIPS")

            # Log extraction results
            logger.info("Extraction Results:")
            logger.info(f"Blog Post extracted: {'Yes' if blog_post else 'No'}")
            logger.info(f"SEO Score extracted: {'Yes' if seo_score else 'No'}")
            logger.info(
                f"Optimization Tips extracted: {'Yes' if optimization_tips else 'No'}"
            )

            return {
                "main_content": blog_post
                or "Error: Failed to generate blog post content",
                "title": content.get("title", ""),
                "meta_description": content.get("meta_description", ""),
                "retrieved_context": retrieved_context,
                "seo_score": seo_score or "Error: Failed to generate SEO score",
                "optimization_tips": optimization_tips
                or "Error: Failed to generate optimization tips",
            }

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            logger.error(f"Traceback:", exc_info=True)
            raise
