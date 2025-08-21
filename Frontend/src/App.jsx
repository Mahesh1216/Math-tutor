import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [feedback, setFeedback] = useState('');
  const [feedbackMessageIndex, setFeedbackMessageIndex] = useState(null);

  const chatWindowRef = useRef(null);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (input.trim() === '') return;

    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input }),
      });
      const data = await response.json();
      const agentMessage = { text: data.answer, sender: 'agent' };
      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      console.error("Error fetching answer:", error);
      const errorMessage = { text: 'Sorry, something went wrong.', sender: 'agent' };
      setMessages(prev => [...prev, errorMessage]);
    }

    setInput('');
  };

  const handleFeedback = async (index) => {
    const initialAnswer = messages[index].text;
    const question = messages[index - 1].text; // Assuming user question precedes agent answer

    try {
      const response = await fetch('http://127.0.0.1:8000/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, initial_answer: initialAnswer, feedback }),
      });
      const data = await response.json();
      const refinedMessage = { text: `(Refined) ${data.refined_answer}`, sender: 'agent' };
      setMessages(prev => [...prev, refinedMessage]);
    } catch (error) {
      console.error("Error sending feedback:", error);
    }

    setFeedback('');
    setFeedbackMessageIndex(null);
  };

  return (
    <div className="app-container">
      <div className="header">Math Tutor Agent</div>
      <div className="chat-window" ref={chatWindowRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <div className="message-bubble">{msg.text}</div>
            {msg.sender === 'agent' && (
              <div className="feedback-section">
                {feedbackMessageIndex === index ? (
                  <>
                    <input
                      type="text"
                      className="feedback-input"
                      value={feedback}
                      onChange={(e) => setFeedback(e.target.value)}
                      placeholder="Provide feedback..."
                    />
                    <button onClick={() => handleFeedback(index)} className="feedback-button">Submit Feedback</button>
                  </>
                ) : (
                  <button onClick={() => setFeedbackMessageIndex(index)} className="feedback-button">Refine</button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          type="text"
          className="input-field"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask a math question..."
        />
        <button onClick={handleSend} className="send-button">Send</button>
      </div>
    </div>
  );
}

export default App;
