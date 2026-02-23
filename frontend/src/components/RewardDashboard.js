import React from 'react';

function RewardDashboard({ userId, score: propScore }) {
  const [score, setScore] = React.useState(propScore || 0);

  React.useEffect(() => {
    if (propScore !== undefined && propScore !== null) {
      setScore(propScore);
    }
  }, [propScore]);
  return (
    <div className="bg-gradient-to-br from-yellow-200 to-orange-200 p-4 rounded-2xl shadow-xl border-4 border-yellow-400 h-full flex flex-col relative overflow-hidden">
      {/* Animated sparkles */}
      <div className="absolute top-0 left-0 text-2xl animate-float" style={{animationDelay: '0s'}}>
        {String.fromCodePoint(0x2728)}
      </div>
      <div className="absolute top-0 right-0 text-2xl animate-float" style={{animationDelay: '0.5s'}}>
        {String.fromCodePoint(0x2B50)}
      </div>
      <div className="absolute bottom-0 left-0 text-2xl animate-float" style={{animationDelay: '1s'}}>
        {String.fromCodePoint(0x1F31F)}
      </div>
      <div className="absolute bottom-0 right-0 text-2xl animate-float" style={{animationDelay: '1.5s'}}>
        {String.fromCodePoint(0x2728)}
      </div>
      
      <h2 className="text-2xl font-bold mb-3 text-orange-800 text-center flex items-center justify-center gap-2 relative z-10">
        <span className="text-3xl animate-bounce">{String.fromCodePoint(0x1F3C6)}</span>
        <span>බෝනස්</span>
        <span className="text-3xl animate-bounce" style={{animationDelay: '0.1s'}}>{String.fromCodePoint(0x1F389)}</span>
      </h2>
      
      <div className="bg-white bg-opacity-90 p-4 rounded-xl shadow-inner flex-1 flex flex-col justify-center relative z-10">
        <p className="text-lg text-gray-600 mb-2 text-center">ඔබේ ලකුණු</p>
        <p className="text-5xl font-bold text-orange-600 mb-2 text-center animate-pulse">
          <span className="inline-block animate-star-bounce">{String.fromCodePoint(0x2B50)}</span> 
          {score} 
          <span className="inline-block animate-star-bounce" style={{animationDelay: '0.2s'}}>{String.fromCodePoint(0x2B50)}</span>
        </p>
        
        <div className="mt-4 text-center">
          {score >= 50 && (
            <div className="text-3xl animate-bounce">
              {String.fromCodePoint(0x1F31F)}{String.fromCodePoint(0x2B50)}{String.fromCodePoint(0x2728)}
            </div>
          )}
          {score >= 100 && (
            <p className="text-lg font-semibold text-purple-600 mt-2 animate-slide-in-up">
              සුපිරි! දිගටම යන්න! {String.fromCodePoint(0x1F4AA)}
            </p>
          )}
          {score < 50 && score > 0 && (
            <p className="text-base text-gray-600 mt-2">
              ඉගෙන ගන්න! {String.fromCodePoint(0x1F4DA)}{String.fromCodePoint(0x270F)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default RewardDashboard;
