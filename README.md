### Financial RAG for Analyzing Quarterly and Annual Revenue Reports of Tech Organizations.
This end to end application was submitted at the MSADS Hackathon 2024 at University of Chicago and secured an Honorable Mention.

### Problem Statement:
In the financial sector, analyzing and summarizing quarterly reports from major organizations is a critical task that requires significant time and expertise. These reports are often dense with financial data, complex narratives, and intricate details, making it challenging for analysts to quickly extract actionable insights. The goal of this project was to develop an automated solution that could efficiently process, analyze, and summarize the quarterly reports of four major organizations, thereby streamlining the decision-making process for stakeholders.

### Solution Overview: 

To address this challenge, I built a financial Retrieval-Augmented Generation (RAG) system using custom agents combined with Google API and open-source Large Language Models (LLMs). This system was designed to automatically analyze and summarize the extensive quarterly reports, extracting key financial insights and presenting them in a concise format. By leveraging the capabilities of RAG and LLMs, the solution not only reduced the time required to process these reports but also improved the accuracy and relevance of the insights generated. The project was successfully implemented and recognized for its innovation, securing an Honorable Mention at the MSADS Hackathon 2024 at the University of Chicago.


### Process:

The technology stack I employed in this project is a carefully curated set of advanced tools designed to handle various aspects of data processing, retrieval, and deployment. For extracting images and tables from unstructured data, I utilized Unstructured.io, a tool known for its capability to efficiently parse and organize complex data formats. To further refine and structure the parsed information, LlamaParse was used, which excels at parsing both text and tables into a format suitable for subsequent analysis.

In handling the conversion of visual data into text, Phi3 Vision played a crucial role by summarizing images into a textual format that could be seamlessly integrated into the data pipeline. To enhance the retrieval-augmented generation (RAG) process, I incorporated Instruct-XL, which provides high-quality embeddings that facilitate more accurate and contextually relevant retrievals. For the generation phase, I selected Qwen2, a large language model with an impressive 32K context window, enabling it to generate detailed and context-rich responses.

To ensure that the retrieval process was as precise as possible, I utilized Col Bert, a state-of-the-art reranker that optimizes retrieval results by refining the order based on relevance and quality. In addition, Llama 3 was employed to generate hypothetical queries, further enriching the dataset and improving the model's robustness. Finally, the entire system was deployed using Streamlit, a user-friendly platform that allows for rapid development and deployment of interactive web applications, ensuring that the end product was both functional and accessible.


### Demo:

https://github.com/user-attachments/assets/07e1bdfa-0f1f-424d-9df8-f72aea90ec7a




