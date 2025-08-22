import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

// SVG Icons
const SendIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const TypingIndicator = () => (
  <div className="typing-indicator">
    <div className="typing-dot"></div>
    <div className="typing-dot"></div>
    <div className="typing-dot"></div>
  </div>
);

function App() {
  const [messages, setMessages] = useState([
    { text: "Hello! I'm your Math Tutor. Ask me any math-related questions, and I'll help you understand the concepts.", sender: 'agent' }
  ]);
  const [input, setInput] = useState('');
  const [feedback, setFeedback] = useState('');
  const [feedbackMessageIndex, setFeedbackMessageIndex] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const chatWindowRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-focus input on load
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = useCallback(async () => {
    const userInput = input.trim();
    if (userInput === '') return;

    // Add user message
    const userMessage = { 
      text: userInput, 
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userInput }),
      });
      
      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      const agentMessage = { 
        text: data.answer, 
        sender: 'agent',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      console.error("Error fetching answer:", error);
      const errorMessage = { 
        text: 'Sorry, I encountered an error while processing your request. Please try again later.', 
        sender: 'agent',
        isError: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [input, messages]);

  const handleFeedback = async (index) => {
    if (!feedback.trim()) {
      setFeedbackMessageIndex(null);
      return;
    }

    const initialAnswer = messages[index].text;
    const question = messages[index - 1]?.text || ''; // Safely get the question
    const feedbackMessage = { 
      text: `Feedback: ${feedback}`, 
      sender: 'user',
      isFeedback: true,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, feedbackMessage]);
    setFeedback('');
    setFeedbackMessageIndex(null);
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question, 
          initial_answer: initialAnswer, 
          feedback: feedback 
        }),
      });
      
      if (!response.ok) throw new Error('Failed to refine answer');
      
      const data = await response.json();
      const refinedMessage = { 
        text: data.refined_answer, 
        sender: 'agent',
        isRefined: true,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, refinedMessage]);
    } catch (error) {
      console.error("Error sending feedback:", error);
      const errorMessage = { 
        text: 'Sorry, I encountered an error while processing your feedback.', 
        sender: 'agent',
        isError: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Format message text with markdown support
  const formatMessage = (text) => {
    if (!text) return '';
    // Convert markdown code blocks
    const withCode = text.replace(/```(\w*)\n([\s\S]*?)\n```/g, 
      (_, lang, code) => `<pre><code class="language-${lang || 'text'}">${code}</code></pre>`
    );
    // Convert line breaks to paragraphs
    return withCode.split('\n').map((p, i) => 
      p.trim() ? `<p>${p}</p>` : ''
    ).join('');
  };

  return (
    <div className="app-container">
      <div className="header">
        <div className="header-content">Math Tutor</div>
      </div>
      
      <div className="chat-window" ref={chatWindowRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <div className="message-content">
              <div className="message-avatar">
                {msg.sender === 'agent' ? 'AI' : 'You'}
              </div>
              <div 
                className="message-bubble"
                dangerouslySetInnerHTML={{ __html: formatMessage(msg.text) }}
              />
            </div>
            {msg.sender === 'agent' && (
              <div className="feedback-section">
                {feedbackMessageIndex === index ? (
                  <div className="feedback-form">
                    <input
                      type="text"
                      className="feedback-input"
                      value={feedback}
                      onChange={(e) => setFeedback(e.target.value)}
                      onKeyDown={(e) => e.key === 'Escape' && setFeedbackMessageIndex(null)}
                      placeholder="Provide feedback..."
                      autoFocus
                    />
                    <div className="feedback-buttons">
                      <button 
                        onClick={() => handleFeedback(index)}
                        className="feedback-button"
                        disabled={!feedback.trim()}
                      >
                        Send
                      </button>
                      <button 
                        onClick={() => {
                          setFeedback('');
                          setFeedbackMessageIndex(null);
                        }}
                        className="feedback-button cancel"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <button 
                    onClick={() => setFeedbackMessageIndex(index)}
                    className="feedback-button"
                    disabled={isLoading}
                  >
                    Refine response
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message agent">
            <div className="message-content">
              <div className="message-avatar">AI</div>
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="input-container">
        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              className="input-field"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask me anything about math..."
              rows="1"
            />
            <button 
              onClick={handleSend} 
              className="send-button"
              disabled={!input.trim() || isLoading}
            >
              <SendIcon />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
