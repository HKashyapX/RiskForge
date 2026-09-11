import React, { useState } from 'react';
import { fetchDemoData } from '../../api/adapters/index';
import { formatPercentage } from '../../utils/formatters';
import { Sparkles, ArrowRight } from 'lucide-react';

const RiskForgeAssistant: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant', content: string }[]>([
    { role: 'assistant', content: "Hello! I'm the RiskForge Assistant prototype. I can query our development dataset to find high-risk incidents, repeating patterns, or explain routing decisions." }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInput('');
    setLoading(true);

    try {
      // Very basic static keyword matching over the dummy data since there's no backend AI
      const allData = await fetchDemoData();
      
      let response = "I couldn't find specific data for that query in the prototype dataset.";

      const lowerInput = userMessage.toLowerCase();
      
      if (lowerInput.includes('escalate') || lowerInput.includes('critical')) {
        const escalations = allData.filter(d => d.inference.routing === 'critical_escalation');
        response = `There are currently ${escalations.length} critical escalations in the dataset. They typically involve high-energy hazards like fires, explosions, or mechanical lifting incidents.`;
      } else if (lowerInput.includes('density') || lowerInput.includes('spd')) {
        const sifPrecursors = allData.filter(d => d.inference.raw_sif_p_score !== null && d.inference.raw_sif_p_score >= 0.40).length;
        const spd = allData.length > 0 ? formatPercentage(sifPrecursors / allData.length * 100) : '0%';
        response = `Currently, the prototype dataset has ${sifPrecursors} SIF precursors out of ${allData.length} total logs, resulting in a SIF Precursor Density (SPD) of ${spd}.`;
      } else if (lowerInput.includes('barrier') || lowerInput.includes('fail')) {
        response = `The dataset currently does not have granular barrier extraction populated. In production, this would list the most frequently bypassed barriers.`;
      } else if (lowerInput.includes('asset') || lowerInput.includes('highest-risk')) {
        response = `Asset information is currently marked as "Not available" in this specific OSHA prototype dataset.`;
      } else if (lowerInput.includes('pattern') || lowerInput.includes('recurring')) {
        response = `Insufficient granular pattern data in this dataset to generate confident analytics.`;
      }

      setTimeout(() => {
        setMessages(prev => [...prev, { role: 'assistant', content: response }]);
        setLoading(false);
      }, 600); // Simulate typing

    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "An error occurred accessing the data adapter." }]);
      setLoading(false);
    }
  };

  return (
    <>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          padding: '12px 24px',
          backgroundColor: '#1e293b', // Dark blue background
          color: 'white',
          border: 'none',
          borderRadius: '30px',
          fontSize: '15px',
          fontWeight: 600,
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          zIndex: 1000
        }}
      >
        <Sparkles size={18} fill="#ea580c" color="#ea580c" />
        Ask RiskForge
        <ArrowRight size={18} />
      </button>

      {isOpen && (
        <div style={{
          position: 'fixed',
          bottom: '80px',
          right: '24px',
          width: '350px',
          height: '500px',
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: '12px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.1)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 1000,
          overflow: 'hidden'
        }}>
          <div style={{ padding: '16px', backgroundColor: 'var(--color-bg-secondary)', borderBottom: '1px solid var(--color-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--color-text-primary)' }}>RiskForge Assistant</div>
            <div style={{ fontSize: '10px', backgroundColor: 'var(--color-warning)', padding: '2px 6px', borderRadius: '4px', color: 'white', fontWeight: 600 }}>PROTOTYPE</div>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {messages.map((msg, idx) => (
              <div key={idx} style={{ 
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                backgroundColor: msg.role === 'user' ? 'var(--color-accent)' : 'var(--color-bg-secondary)',
                color: msg.role === 'user' ? 'white' : 'var(--color-text-primary)',
                padding: '12px 16px',
                borderRadius: '8px',
                maxWidth: '85%',
                fontSize: '13px',
                lineHeight: 1.5
              }}>
                {msg.content}
              </div>
            ))}
            {loading && (
              <div style={{ alignSelf: 'flex-start', fontSize: '12px', color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
                Querying development data...
              </div>
            )}
          </div>

          <div style={{ padding: '16px', borderTop: '1px solid var(--color-border)', display: 'flex', gap: '8px' }}>
            <input 
              type="text" 
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              placeholder="Ask about incidents, density..."
              style={{ flex: 1, padding: '8px 12px', borderRadius: '4px', border: '1px solid var(--color-border)', fontSize: '13px' }}
            />
            <button 
              onClick={handleSend}
              disabled={loading || !input.trim()}
              style={{ padding: '8px 16px', backgroundColor: 'var(--color-text-primary)', color: 'white', border: 'none', borderRadius: '4px', fontSize: '13px', cursor: 'pointer', fontWeight: 600 }}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default RiskForgeAssistant;
