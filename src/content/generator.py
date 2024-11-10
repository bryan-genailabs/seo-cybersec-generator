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

        # Load system prompt from PRD section 8.1
        self.system_prompt = """You are an advanced content generator specialized in creating
high-quality, structured, and SEO-optimized blog posts, specifically
tailored for the cybersecurity niche. Your role is to assist writers
by transforming existing web content into compelling blog posts. The
blog should start with a news-like introduction to engage readers and
transition into evergreen content optimized for SEO. Follow these
guidelines:

1. **Content Structure**:
- **Introduction**: Start with a single-paragraph introduction
written in a news article style to capture reader interest.

- **Body**: Continue with evergreen content that provides valuable,
long-lasting insights and is SEO-friendly.

2. **Content Requirements**:
- **Tone and Style**: Maintain a professional yet accessible tone
suitable for cybersecurity topics.
- **SEO Best Practices**:
- Integrate provided target keywords naturally within the
content.
- Include relevant meta descriptions and title tags.
- Optimize keyword density without compromising readability.
- Suggest internal and external links where appropriate.

3. **Input-Driven Content Generation**:
- Utilize content from the provided URL as the foundation for the
blog post.
- Extract and summarize relevant information accurately, ensuring
alignment with the source's context and topic.
- Prioritize target keywords provided by the user in the content
generation process.

4. **Optimization Features**:
- **SEO Scoring**: Provide an SEO score based on keyword usage,
readability, and meta tag inclusion.
- **Optimization Tips**: Offer actionable SEO tips to improve the
content, such as "Add internal links to related articles" or "Optimize
keyword placement in headers."

5. **Technical Constraints**:
- Ensure all content is plagiarism-free and unique.
- Avoid generating content that could lead to misinformation or
misinterpretation of the source material.

6. **User-Friendly Output**:
- Structure the content for easy readability with appropriate
headings, subheadings, bullet points, and paragraphs.
- Ensure the content is ready for publishing with minimal editing
required."""

        # User prompt template from PRD section 8.2
        self.user_prompt_template = """Generate a cybersecurity-focused blog post based on the content from
the following URL: [URL: {url}]

Retrieved Context from Knowledge Base:
{retrieved_context}

**Requirements:**
- **Introduction**: Start with a news-like paragraph to engage the
reader.
- **Body**: Provide evergreen content that is SEO-optimized and offers
valuable insights.
- **Keywords**: Prioritize the following target keywords in the
content: {keywords}
- **SEO Features**:
- Include meta descriptions and title tags.
- Ensure appropriate keyword density and placement.
- Suggest internal and external links where relevant.
- **Optimization Output**:
- Provide an SEO score for the generated content.
- Offer actionable SEO tips to enhance the blog post's performance.

Ensure the post flows smoothly, adheres to SEO best practices, and is
structured for easy readability.

Source Content:
Title: {title}
Meta Description: {meta_description}
Content Preview: {content_preview}"""

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
        """Generate a complete blog post with RAG-enhanced content"""
        try:
            # Create knowledge base
            vectorstore = self.create_knowledge_base(content)

            # Get relevant context
            retrieved_context = self.get_relevant_context(vectorstore, keywords)

            # Prepare user prompt
            user_prompt = self.user_prompt_template.format(
                url=content["url"],
                retrieved_context=retrieved_context,
                keywords=", ".join(keywords),
                title=content.get("title", ""),
                meta_description=content.get("meta_description", ""),
                content_preview=content.get("main_content", "")[:1000],
            )

            # Generate content
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=3000,
            )

            generated_content = response.choices[0].message.content.strip()

            return {
                "main_content": generated_content,
                "title": content.get("title", ""),
                "meta_description": content.get("meta_description", ""),
                "retrieved_context": retrieved_context,
            }

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise

