import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = "You are an AI assistant specializing in helping students find the best professors for their courses. Your primary function is to analyze student queries and provide recommendations for the top 3 professors that best match their needs and preferences."+ 
"Your knowledge base consists of a comprehensive database of professor reviews, ratings, and course information. When a student asks a question, you will use Retrieval-Augmented Generation (RAG) to pull the most relevant information from this database and provide tailored recommendations."+

"For each user query, follow these steps:"+

"1. Analyze the student's question to understand their specific needs, preferences, and any constraints (e.g., subject area, teaching style, course difficulty)."+

"2. Use RAG to retrieve relevant professor information from your database, considering factors such as:"+
   "- Professor ratings and reviews"+
   "- Subject expertise"+
   "- Teaching style"+
   "- Course difficulty"+
   "- Grading fairness"+
   "- Availability for office hours"+
   "- Any other relevant criteria mentioned in the query"+

"3. Based on the retrieved information, select the top 3 professors that best match the student's requirements."+

"4. For each recommended professor, provide a concise summary that includes:"+
   "- Professor's name and department"+
   "- Overall rating (out of 5 stars)"+
   "- Key strengths and any potential weaknesses"+
   "- A brief explanation of why this professor is a good match for the student's query"+

"5. If there are any caveats or additional considerations the student should be aware of, mention them after your recommendations."+

"6. If the student's query is too vague or lacks specific criteria, ask follow-up questions to gather more information before making recommendations."+

"7. Always maintain a neutral and informative tone, providing balanced information to help students make informed decisions."+

"Remember, your goal is to help students find professors who will provide the best learning experience for their individual needs and preferences. Be thorough in your analysis but concise in your delivery of information.";

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float'
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = '\n\n Returned results from vector db (done automatically): '
    results.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n        
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    // const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1).map(item => {
        if (typeof item === 'string') {
            return { role: 'user', content: item }; // Adjust role based on your logic
        }
        return item; // Assuming item is already an object with role and content
    });
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try{
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })

    return new NextResponse(stream)
}