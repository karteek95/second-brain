import React, { useState, useRef, useEffect } from 'react';
import {
  Container,
  TextField,
  IconButton,
  Typography,
  Box,
  Paper,
  Avatar
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';

export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Automatically scroll down when new messages or chunks arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userText = input.trim();
    setInput('');
    setLoading(true);

    // 1. Add user message
    const userMsg = { id: Date.now().toString(), role: 'user', content: userText };

    // 2. Add an empty placeholder for the AI's streaming response
    const aiMsgId = (Date.now() + 1).toString();
    const aiMsgPlaceholder = { id: aiMsgId, role: 'assistant', content: '' };

    setMessages((prev) => [...prev, userMsg, aiMsgPlaceholder]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userText }),
      });

      if (!response.body) throw new Error('ReadableStream not supported.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      // 3. Stream chunks directly into the specific AI message object
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMsgId
              ? { ...msg, content: msg.content + chunk }
              : msg
          )
        );
      }
    } catch (error) {
      console.error('Error fetching stream:', error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMsgId
            ? { ...msg, content: 'An error occurred while fetching the answer.' }
            : msg
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container
      maxWidth="md"
      sx={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        py: 2
      }}
    >
      {/* Header */}
      <Typography variant="h5" align="center" gutterBottom sx={{ fontWeight: 'bold', color: '#333' }}>
        Knowledge Base Search
      </Typography>

      {/* Chat History Area */}
      <Box
        sx={{
          flexGrow: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          p: 2,
          mb: 2,
          bgcolor: '#fafafa',
          borderRadius: 2
        }}
      >
        {messages.length === 0 && (
          <Typography variant="body1" align="center" color="textSecondary" sx={{ mt: 'auto', mb: 'auto' }}>
            What can I help you with today?
          </Typography>
        )}

        {messages.map((msg) => (
          <Box
            key={msg.id}
            sx={{
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              gap: 1.5
            }}
          >
            {msg.role === 'assistant' && (
              <Avatar sx={{ bgcolor: '#1976d2', width: 32, height: 32 }}>
                <SmartToyIcon fontSize="small" />
              </Avatar>
            )}

            <Paper
              elevation={0}
              sx={{
                p: 2,
                maxWidth: '75%',
                bgcolor: msg.role === 'user' ? '#e3f2fd' : '#ffffff',
                border: '1px solid',
                borderColor: msg.role === 'user' ? '#bbdefb' : '#e0e0e0',
                borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px'
              }}
            >
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                {msg.content}
              </Typography>
            </Paper>

            {msg.role === 'user' && (
              <Avatar sx={{ bgcolor: '#9c27b0', width: 32, height: 32 }}>
                <PersonIcon fontSize="small" />
              </Avatar>
            )}
          </Box>
        ))}
        {/* Invisible div to anchor our scrolling */}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Paper
        elevation={3}
        sx={{
          p: '4px 8px',
          display: 'flex',
          alignItems: 'center',
          borderRadius: 8,
          border: '1px solid #e0e0e0'
        }}
      >
        <TextField
          fullWidth
          placeholder="Message the agent..."
          variant="standard"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Allow Shift+Enter for new lines, but Enter alone sends the message
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={loading}
          multiline
          maxRows={5}
          InputProps={{
            disableUnderline: true,
            sx: { ml: 2, flex: 1, py: 1 }
          }}
        />
        <IconButton
          color="primary"
          sx={{ p: '10px' }}
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          <SendIcon />
        </IconButton>
      </Paper>
    </Container>
  );
}