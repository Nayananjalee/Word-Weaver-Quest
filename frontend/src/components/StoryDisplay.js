import React from 'react';

function StoryDisplay({ story, question, options, onAnswer }) {
  return (
    <div className="bg-gradient-to-br from-yellow-100 to-pink-100 p-8 rounded-3xl shadow-2xl border-4 border-purple-400 relative overflow-hidden transform hover:scale-105 transition-transform duration-300">
      {/* Decorative elements */}
      <div className="absolute top-2 right-2 text-4xl animate-bounce">📚</div>
      <div className="absolute bottom-2 left-2 text-4xl animate-pulse">🎨</div>
      
      <div className="relative z-10">
        <h2 className="text-3xl font-bold mb-6 text-purple-700 flex items-center justify-center gap-2">
          <span>🌟</span> කතාව <span>🌟</span>
        </h2>
        <div className="bg-white bg-opacity-80 p-6 rounded-2xl mb-6 shadow-inner">
          <p className="text-xl leading-relaxed text-gray-800 font-medium">{story}</p>
        </div>
        
        <div className="bg-gradient-to-r from-blue-200 to-purple-200 p-6 rounded-2xl shadow-lg">
          <h3 className="text-2xl font-bold mb-4 text-purple-800 flex items-center justify-center gap-2">
            <span>🤔</span> {question}
          </h3>
          <div className="flex flex-col gap-4">
            {options.map((option, index) => (
              <button
                key={index}
                onClick={() => onAnswer(option)}
                className="bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 text-white font-bold py-4 px-6 rounded-full text-lg shadow-lg transform hover:scale-105 active:scale-95 transition-all duration-200 border-2 border-white"
              >
                <span className="flex items-center justify-center gap-2">
                  <span className="text-2xl">{['🅰️', '🅱️', '🅲️'][index]}</span>
                  {option}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default StoryDisplay;
