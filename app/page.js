'use client'
import Image from "next/image";
import { useState } from "react"
import {
  Stack,
  Box,
  TextField,
  Button,  
} from '@mui/material'
// import { League_Gothic } from "next/font/google";

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi! I'm the Rate My Professor assistant. How can I help you today?"
    }
  ])
const [message, setMessage] = useState('')
const sendMessage = async () => {
  setMessages((messages)=> [
    ...messages,
    {role: "user", content: message},
    {role: "assistant", content: ''}
 ])

 setMessage('')
try {
  const response = await fetch('/api/chat', {
    method: "POST",
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify([...messages, {role: "user", content: message}]),
   }) 
  
   if (!response.ok) {
    // Handle error response
    throw new Error(`HTTP error! Status: ${response.status}`);
  }
  
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
  
    let result = ''
    return reader.read().then(function processText({done, value}) {
      if (done) {
        return result
      }
      const text = decoder.decode(value || new Uint8Array(), {stream: true})
      
      setMessages((messages)=> {
        let lastMessage = messages[messages.length - 1]
        let otherMessages = messages.slice(0, messages.length - 1)
        return [
          ...otherMessages,
          {...lastMessage, content: lastMessage.content + text},
        ]
      })
      return reader.read().then(processText)
    })
} catch (error) {
  console.error('Failed to fetch:', error);
}
}

return (
  <Box 
  width="100vw"
  height="100vh"
  display="flex"
  flexDirection="column"
  justifyContent="center"
  alignItems="center"
  >
    <Stack 
    direction="column"
    width="500px"
    height="700px"
    border="1px solid black"
    p={2}
    spacing={3}
    >
      <Stack 
          direction="column"
          spacing={2}
          flexGrow={1}
          overflow='auto'
          maxHeight='100%'
          >
      {messages.map((message, index)=> (
        <Box
        key={index}
        display="flex"
        justifyContent={
          MessageChannel.ROLE === 'assistant' ? 'flex-start' : 'flex-end'
        }
        >
        <Box 
        bgcolor={
          message.role === 'assistant' ? 'primary.main' : 'secondary.main'
        }
        color = "white"
        borderRadius={16}
        p={3}
        >
          {message.content}
        </Box>        
        </Box>
      ))}
    </Stack>
    <Stack 
    direction = "row" spacing = {2}>
      <TextField
      label = "Message"
      fullWidth
      value={message}
      onChange={(e)=> {
        setMessage(e.target.value)
      }} />

      <Button variant='contained' onClick={sendMessage}>
        Send
      </Button>
    </Stack>
    </Stack>
  </Box>
    
  )
}
