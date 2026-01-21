import re
from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np


class FaithfulnessScorer:
    
    def __init__(self, use_embeddings=True, llm=None):
        self.use_embeddings = use_embeddings
        self.embedder = None
        self.llm = None
        
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except:
                print("⚠ Using text-only ")
                self.use_embeddings = False
        else:
            print("✓ Text-only ")
    
    def extract_claims(self, answer: str) -> List[str]:
        answer = re.sub(r'(Question:|Answer:|Step \d+:)\s*', '', answer)
        answer = re.sub(r'\n{3,}', '\n', answer)
        
        if "cannot answer" in answer.lower() or len(answer.strip()) < 20:
            return []
        
        sentences = re.split(r'[.!?]+', answer)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 4:
                words = sent.lower().split()
                if len(words) > 0 and len(set(words)) / len(words) >= 0.30:
                    claims.append(sent)
        
        return claims
    
    def compute_rouge_l(self, claim: str, context: str) -> float:
        def lcs_length(s1: List[str], s2: List[str]) -> int:
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1].lower() == s2[j-1].lower():
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        claim_words = claim.split()
        context_words = context.split()
        
        if not claim_words:
            return 1.0
        
        lcs_len = lcs_length(claim_words, context_words)
        recall = lcs_len / len(claim_words)
        precision = lcs_len / len(context_words) if context_words else 0
        
        if recall + precision == 0:
            return 0.0
        
        return float(2 * (precision * recall) / (precision + recall))
    
    def compute_rouge_2(self, claim: str, context: str) -> float:
        def get_bigrams(text: str):
            words = text.lower().split()
            return [tuple(words[i:i+2]) for i in range(len(words) - 1)]
        
        claim_bigrams = get_bigrams(claim)
        if not claim_bigrams:
            return 1.0
        
        context_bigrams = get_bigrams(context)
        claim_counter = Counter(claim_bigrams)
        context_counter = Counter(context_bigrams)
        
        overlap = sum(min(claim_counter[bg], context_counter[bg]) for bg in claim_counter)
        return float(overlap / sum(claim_counter.values()))
    
    def compute_bleu(self, claim: str, context: str) -> float:
        def get_ngrams(text: str, n: int):
            words = text.lower().split()
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        precisions = []
        for n in range(1, 5):
            claim_ngrams = get_ngrams(claim, n)
            if not claim_ngrams:
                continue
            
            context_ngrams = get_ngrams(context, n)
            claim_counter = Counter(claim_ngrams)
            context_counter = Counter(context_ngrams)
            
            overlap = sum(min(claim_counter[ng], context_counter[ng]) for ng in claim_counter)
            precisions.append(overlap / len(claim_ngrams))
        
        if not precisions:
            return 1.0
        
        return float(np.exp(np.mean(np.log(np.array(precisions) + 1e-10))))
    
    def compute_semantic_similarity(self, claim: str, context: str) -> float:
        if not self.use_embeddings or self.embedder is None:
            return self._word_overlap(claim, context)
        
        try:
            claim_emb = self.embedder.encode(claim, convert_to_tensor=False)
            
            sentences = re.split(r'[.!?]+', context)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                sentences = [context]
            
            max_sim = 0.0
            for sent in sentences[:30]:
                sent_emb = self.embedder.encode(sent, convert_to_tensor=False)
                sim = np.dot(claim_emb, sent_emb) / (
                    np.linalg.norm(claim_emb) * np.linalg.norm(sent_emb) + 1e-10
                )
                max_sim = max(max_sim, float(sim))
            
            return max_sim
        except:
            return self._word_overlap(claim, context)
    
    def _word_overlap(self, claim: str, context: str) -> float:
        def stem(w):
            w = w.lower()
            for suf in ['ization', 'isation', 'ation', 'tion', 'ing', 'ed', 's', 'es', 'ly', 'er', 'est']:
                if w.endswith(suf) and len(w) > len(suf) + 3:
                    return w[:-len(suf)]
            return w
        
        stops = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by'}
        
        claim_words = [stem(w) for w in claim.lower().split() if w not in stops]
        context_words = [stem(w) for w in context.lower().split()]
        
        if not claim_words:
            return 1.0
        
        matches = sum(1 for w in claim_words if w in context_words)
        return matches / len(claim_words)
    
    def verify_claim(self, claim: str, context: str) -> Dict[str, Any]:
        rouge_l = self.compute_rouge_l(claim, context)
        rouge_2 = self.compute_rouge_2(claim, context)
        bleu = self.compute_bleu(claim, context)
        semantic = self.compute_semantic_similarity(claim, context)
        word_overlap = self._word_overlap(claim, context)
        
        if self.embedder is not None:
            combined = (
                0.40 * semantic +      # Higher weight on semantic
                0.25 * rouge_l +
                0.15 * rouge_2 +
                0.10 * bleu +
                0.10 * word_overlap
            )
            threshold = 0.18
        else:
            combined = (
                0.40 * rouge_l +       # Higher ROUGE-L weight
                0.25 * rouge_2 +
                0.15 * bleu +
                0.20 * word_overlap
            )
            threshold = 0.15
        
        return {
            'supported': combined >= threshold,
            'confidence': combined,
            'metrics': {
                'rouge_l': float(rouge_l),
                'rouge_2': float(rouge_2),
                'bleu': float(bleu),
                'semantic_similarity': float(semantic),
                'word_overlap': float(word_overlap),
                'combined': float(combined),
                'threshold': threshold
            }
        }
    
    def compute_faithfulness_score(self, answer: str, context: str) -> Dict[str, Any]:
        claims = self.extract_claims(answer)
        
        if not claims:
            if "cannot answer" in answer.lower():
                return {
                    'faithfulness_score': 1.0,
                    'num_claims': 0,
                    'supported_claims': 0,
                    'unsupported_claims': [],
                    'verdict': 'ABSTAIN',
                    'confidence': 1.0
                }
            return {
                'faithfulness_score': 0.0,
                'num_claims': 0,
                'supported_claims': 0,
                'unsupported_claims': [],
                'verdict': 'FAIL',
                'confidence': 0.0
            }
        
        supported_count = 0
        unsupported_claims = []
        claim_details = []
        
        for claim in claims:
            result = self.verify_claim(claim, context)
            
            claim_details.append({
                'claim': claim,
                'supported': result['supported'],
                'confidence': result['confidence'],
                'metrics': result['metrics']
            })
            
            if result['supported']:
                supported_count += 1
            else:
                unsupported_claims.append(claim)
        
        faithfulness = supported_count / len(claims)
        avg_confidence = sum(c['confidence'] for c in claim_details) / len(claims)
        pass_threshold = 0.40 if self.embedder else 0.35
        
        if faithfulness >= pass_threshold:
            verdict = 'PASS'
        elif faithfulness >= 0.25:
            verdict = 'PARTIAL'
        else:
            verdict = 'FAIL'
        
        return {
            'faithfulness_score': faithfulness,
            'num_claims': len(claims),
            'supported_claims': supported_count,
            'unsupported_claims': unsupported_claims,
            'claim_details': claim_details,
            'verdict': verdict,
            'confidence': avg_confidence
        }
