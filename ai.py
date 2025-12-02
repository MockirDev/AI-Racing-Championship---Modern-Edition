import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import math

# Deneyimlerin saklanacağı veri yapısı
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """
    Geçmiş deneyimleri saklamak için kullanılan hafıza yapısı.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Bir deneyimi hafızaya kaydeder."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Hafızadan rastgele bir grup deneyim örneği alır."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Derin Q-Network (DQN) sinir ağı modeli.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        İleri yayılım fonksiyonu. Durum (state) girdisini alır
        ve her eylem için Q-değerlerini döndürür.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    """
    Yapay zeka ajanı. Karar verme ve öğrenme süreçlerini yönetir.
    """
    def __init__(self, n_observations, n_actions, difficulty="medium", agent_id=0, behavior_type="balanced"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = agent_id
        
        # İki sinir ağı: biri kararlar için, diğeri öğrenme hedeflerini sabitlemek için
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network'ü sadece değerlendirme modunda kullan

        # AI Difficulty System
        self.difficulty = difficulty.lower()
        difficulty_settings = self._get_difficulty_settings()
        
        # AI Behavioral Patterns System
        self.behavior_type = behavior_type
        self.behavioral_params = self._get_behavioral_params()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=difficulty_settings['learning_rate'], amsgrad=True)
        self.memory = ReplayMemory(difficulty_settings['memory_capacity'])
        
        self.n_actions = n_actions
        self.steps_done = 0
        
        # Epsilon-greedy stratejisi için parametreler (zorluk seviyesine göre ayarlandı)
        self.eps_start = difficulty_settings['eps_start']
        self.eps_end = difficulty_settings['eps_end']
        self.eps_decay = difficulty_settings['eps_decay']
        self.skill_modifier = difficulty_settings['skill_modifier']
        
        # Multi-Agent Learning System
        self.shared_experiences = []  # Diğer ajanlardan gelen deneyimler
        self.cooperation_factor = 0.1  # Diğer ajanlardan öğrenme oranı
        self.competition_mode = True
        self.rival_performance = {}  # Rakiplerin performans bilgileri
        
        # === AI STATISTICS & LEARNING TRACKING ===
        self.stats = {
            'total_rewards': [],
            'episode_rewards': 0.0,
            'episodes_completed': 0,
            'laps_completed': 0,
            'best_lap_reward': -float('inf'),
            'average_reward_last_100': 0.0,
            'exploration_rate': 0.0,
            'learning_progress': []
        }
        
        # Performance tracking
        self.episode_start_time = 0
        self.episode_steps = 0
        self.total_training_steps = 0

    def _get_difficulty_settings(self):
        """
        AI zorluk seviyelerine göre parametreleri döndür
        """
        difficulties = {
            "easy": {
                "learning_rate": 5e-5,      # Yavaş öğrenme
                "memory_capacity": 10000,    # Düşük hafıza
                "eps_start": 0.95,          # Çok exploration
                "eps_end": 0.15,            # Yüksek minimum exploration
                "eps_decay": 3000,          # Hızlı decay
                "skill_modifier": 0.7       # Düşük beceri
            },
            "medium": {
                "learning_rate": 1e-4,      # Normal öğrenme
                "memory_capacity": 20000,    # Normal hafıza
                "eps_start": 0.9,           # Normal exploration
                "eps_end": 0.05,            # Normal minimum exploration
                "eps_decay": 5000,          # Normal decay
                "skill_modifier": 1.0       # Normal beceri
            },
            "hard": {
                "learning_rate": 2e-4,      # Hızlı öğrenme
                "memory_capacity": 30000,    # Yüksek hafıza
                "eps_start": 0.8,           # Az exploration
                "eps_end": 0.02,            # Düşük minimum exploration
                "eps_decay": 7000,          # Yavaş decay
                "skill_modifier": 1.3       # Yüksek beceri
            },
            "expert": {
                "learning_rate": 3e-4,      # Çok hızlı öğrenme
                "memory_capacity": 50000,    # Maksimum hafıza
                "eps_start": 0.7,           # Minimum exploration
                "eps_end": 0.01,            # Çok düşük minimum exploration
                "eps_decay": 10000,         # Çok yavaş decay
                "skill_modifier": 1.5       # Maksimum beceri
            }
        }
        
        return difficulties.get(self.difficulty, difficulties["medium"])

    def _get_behavioral_params(self):
        """
        AI davranış tipine göre parametreleri döndür
        """
        behaviors = {
            "aggressive": {
                "risk_tolerance": 0.8,      # Yüksek risk alma
                "speed_preference": 1.3,    # Hız odaklı
                "overtaking_tendency": 0.9, # Sollama eğilimi
                "exploration_bonus": 1.2,   # Daha fazla keşif
                "patience": 0.3             # Düşük sabır
            },
            "balanced": {
                "risk_tolerance": 0.5,      # Orta risk
                "speed_preference": 1.0,    # Normal hız
                "overtaking_tendency": 0.5, # Normal sollama
                "exploration_bonus": 1.0,   # Normal keşif
                "patience": 0.6             # Orta sabır
            },
            "defensive": {
                "risk_tolerance": 0.2,      # Düşük risk
                "speed_preference": 0.8,    # Güvenli hız
                "overtaking_tendency": 0.2, # Nadir sollama
                "exploration_bonus": 0.8,   # Az keşif
                "patience": 0.9             # Yüksek sabır
            },
            "adaptive": {
                "risk_tolerance": 0.6,      # Duruma göre değişken
                "speed_preference": 1.1,    # Adaptif hız
                "overtaking_tendency": 0.7, # Duruma göre sollama
                "exploration_bonus": 1.1,   # Adaptif keşif
                "patience": 0.5             # Orta sabır
            }
        }
        
        return behaviors.get(self.behavior_type, behaviors["balanced"])

    def share_experience(self, other_agents):
        """
        Diğer ajanlarla deneyim paylaşımı yapar (Multi-Agent Learning)
        """
        if not self.competition_mode and len(other_agents) > 0:
            # En iyi performansa sahip ajanı bul
            best_agent = max(other_agents, key=lambda agent: agent.stats['average_reward_last_100'])
            
            if best_agent.stats['average_reward_last_100'] > self.stats['average_reward_last_100']:
                # En iyi ajandan deneyim kopyala
                if len(best_agent.memory) > 100:
                    shared_transitions = random.sample(list(best_agent.memory.memory), 
                                                     min(50, len(best_agent.memory)))
                    
                    for transition in shared_transitions:
                        if random.random() < self.cooperation_factor:
                            self.memory.push(*transition)

    def update_rival_performance(self, other_agents):
        """
        Rakip ajanların performansını takip eder
        """
        for agent in other_agents:
            if agent.agent_id != self.agent_id:
                self.rival_performance[agent.agent_id] = {
                    'avg_reward': agent.stats['average_reward_last_100'],
                    'laps_completed': agent.stats['laps_completed'],
                    'behavior_type': agent.behavior_type,
                    'difficulty': agent.difficulty
                }

    def get_behavioral_action_modifier(self, state, base_action_probs):
        """
        Davranış tipine göre aksiyon seçimini modifiye eder
        """
        behavior = self.behavioral_params
        
        # State'ten hız ve sensör bilgilerini çıkar
        speed = state[0, 5].item() if len(state[0]) > 5 else 0.5
        min_sensor = min(state[0, :5]).item() if len(state[0]) >= 5 else 1.0
        
        # Davranışsal modifikasyonlar
        action_modifiers = torch.ones_like(base_action_probs)
        
        # Agresif davranış: Hızlı gitmek için ileri aksiyonunu artır
        if self.behavior_type == "aggressive":
            action_modifiers[0] *= (1.0 + behavior["speed_preference"] * 0.3)
            if min_sensor > 0.3:  # Yeterli alan varsa daha riskli
                action_modifiers[0] *= 1.2
        
        # Defansif davranış: Güvenli mesafe koruma
        elif self.behavior_type == "defensive":
            if min_sensor < 0.4:  # Yakın engel varsa yavaşla
                action_modifiers[0] *= 0.7
            # Daha fazla dönüş tercihi (güvenli geçiş)
            action_modifiers[1] *= 1.1
            action_modifiers[2] *= 1.1
        
        # Adaptif davranış: Duruma göre değişken
        elif self.behavior_type == "adaptive":
            if min_sensor < 0.3:
                action_modifiers[1:] *= 1.3  # Dönüşleri artır
            elif speed < 0.7:
                action_modifiers[0] *= 1.4   # Hızlanmayı artır
        
        return action_modifiers

    def select_action(self, state, other_agents=None):
        """
        Gelişmiş epsilon-greedy stratejisi + behavioral patterns
        """
        # Diğer ajanların performansını güncelle
        if other_agents:
            self.update_rival_performance(other_agents)
        
        sample = random.random()
        # Epsilon değeri, zamanla azalır ve davranış tipine göre modifiye edilir
        base_eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        
        # Davranış tipine göre exploration modifikasyonu
        exploration_modifier = self.behavioral_params["exploration_bonus"]
        eps_threshold = base_eps_threshold * exploration_modifier
        
        self.steps_done += 1
        
        if sample > eps_threshold:
            # En iyi bilinen eylemi seç (exploitation) + behavioral modification
            with torch.no_grad():
                q_values = self.policy_net(state)
                
                # Davranışsal modifikasyon uygula
                behavioral_modifiers = self.get_behavioral_action_modifier(state, q_values[0])
                modified_q_values = q_values[0] * behavioral_modifiers
                
                # Skill modifier uygula (zorluk seviyesine göre)
                if self.skill_modifier < 1.0:
                    # Düşük skill - bazen yanlış karar ver
                    if random.random() > self.skill_modifier:
                        return torch.tensor([[random.randrange(self.n_actions)]], 
                                           device=self.device, dtype=torch.long)
                
                return modified_q_values.max(0)[1].view(1, 1)
        else:
            # Behavioral pattern'e göre exploration
            if self.behavior_type == "aggressive":
                # Agresif: Forward bias
                action_weights = [0.6, 0.2, 0.2]  # Forward, Left, Right
            elif self.behavior_type == "defensive":
                # Defansif: Turning bias
                action_weights = [0.3, 0.35, 0.35]  # Forward, Left, Right
            else:
                # Balanced/Adaptive: Uniform
                action_weights = [0.33, 0.33, 0.34]  # Forward, Left, Right
            
            # Weighted random selection
            action = random.choices(range(self.n_actions), weights=action_weights, k=1)[0]
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def adaptive_difficulty_adjustment(self, performance_metrics):
        """
        Performansa göre zorluk seviyesini otomatik ayarlar
        """
        avg_reward = performance_metrics.get('avg_reward', 0)
        win_rate = performance_metrics.get('win_rate', 0)
        
        # Performans çok iyi - zorluğu artır
        if avg_reward > 80 and win_rate > 0.7:
            if self.difficulty != "expert":
                self.eps_end = max(0.01, self.eps_end * 0.8)
                self.skill_modifier = min(1.5, self.skill_modifier * 1.1)
                print(f"AI {self.agent_id} zorluk artırıldı!")
        
        # Performans çok kötü - zorluğu azalt
        elif avg_reward < 20 and win_rate < 0.2:
            if self.difficulty != "easy":
                self.eps_end = min(0.15, self.eps_end * 1.2)
                self.skill_modifier = max(0.7, self.skill_modifier * 0.9)
                print(f"AI {self.agent_id} zorluk azaltıldı!")

    def optimize_model(self, batch_size=128, gamma=0.99):
        """
        Hafızadan alınan bir grup deneyim üzerinden modeli eğitir.
        """
        if len(self.memory) < batch_size:
            return # Yeterli deneyim birikmediyse öğrenme yapma

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Sonraki durumların bir maskesini oluştur (final olmayan durumlar için)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Mevcut durumlar için modelin tahmin ettiği Q-değerleri
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Sonraki durumlar için hedef Q-değerleri
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Beklenen Q-değerlerini hesapla (Bellman denklemi)
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Kayıp (loss) fonksiyonunu hesapla
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Geri yayılım ile modeli güncelle
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """
        Hedef ağı, politika ağının ağırlıklarıyla günceller.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_stats(self, reward, lap_completed=False):
        """
        AI öğrenme istatistiklerini güncelle
        """
        self.stats['episode_rewards'] += reward
        self.episode_steps += 1
        self.total_training_steps += 1
        
        # Exploration rate güncelle
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.stats['exploration_rate'] = eps_threshold
        
        if lap_completed:
            self.stats['laps_completed'] += 1
            
            # Episode tamamlandı
            self.stats['episodes_completed'] += 1
            self.stats['total_rewards'].append(self.stats['episode_rewards'])
            
            # Best lap reward tracking
            if self.stats['episode_rewards'] > self.stats['best_lap_reward']:
                self.stats['best_lap_reward'] = self.stats['episode_rewards']
            
            # Average reward calculation (last 100 episodes)
            recent_rewards = self.stats['total_rewards'][-100:]
            self.stats['average_reward_last_100'] = sum(recent_rewards) / len(recent_rewards)
            
            # Learning progress tracking
            if len(self.stats['total_rewards']) % 10 == 0:  # Every 10 episodes
                progress_point = {
                    'episode': self.stats['episodes_completed'],
                    'avg_reward': self.stats['average_reward_last_100'],
                    'exploration_rate': eps_threshold,
                    'memory_size': len(self.memory)
                }
                self.stats['learning_progress'].append(progress_point)
            
            # Reset episode tracking
            self.stats['episode_rewards'] = 0.0
            self.episode_steps = 0
    
    def get_learning_stats(self):
        """
        AI öğrenme istatistiklerini döndür
        """
        return {
            'episodes': self.stats['episodes_completed'],
            'laps': self.stats['laps_completed'],
            'avg_reward': self.stats['average_reward_last_100'],
            'best_reward': self.stats['best_lap_reward'],
            'exploration': self.stats['exploration_rate'],
            'memory_usage': len(self.memory),
            'total_steps': self.total_training_steps
        }
    
    def save_model(self, filepath):
        """
        AI modelini ve istatistiklerini kaydet
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'stats': self.stats,
            'steps_done': self.steps_done,
            'total_training_steps': self.total_training_steps,
            'episode_steps': self.episode_steps
        }
        
        torch.save(save_data, filepath)
        print(f"AI model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        AI modelini ve istatistiklerini yükle
        """
        try:
            save_data = torch.load(filepath, map_location=self.device)
            
            self.policy_net.load_state_dict(save_data['policy_net_state'])
            self.target_net.load_state_dict(save_data['target_net_state'])
            self.optimizer.load_state_dict(save_data['optimizer_state'])
            self.stats = save_data['stats']
            self.steps_done = save_data['steps_done']
            self.total_training_steps = save_data['total_training_steps']
            self.episode_steps = save_data['episode_steps']
            
            print(f"AI model loaded from: {filepath}")
            print(f"Episodes: {self.stats['episodes_completed']}, Avg Reward: {self.stats['average_reward_last_100']:.2f}")
            return True
            
        except Exception as e:
            print(f"Error loading AI model: {e}")
            return False


class MultiAgentManager:
    """
    Birden fazla AI ajanını yöneten ve koordine eden sistem
    """
    def __init__(self):
        self.agents = []
        self.competition_results = []
        self.training_session_id = 0
        
        # Global learning settings
        self.global_learning_enabled = True
        self.experience_sharing_interval = 100  # Steps between sharing
        self.last_sharing_step = 0
        
        # Competition tracking
        self.race_winners = []
        self.performance_history = []
        
        # Behavioral diversity enforcement
        self.behavioral_balance = {
            "aggressive": 0,
            "balanced": 0, 
            "defensive": 0,
            "adaptive": 0
        }

    def add_agent(self, agent):
        """Sisteme yeni bir ajan ekler"""
        self.agents.append(agent)
        self.behavioral_balance[agent.behavior_type] += 1
        print(f"Agent {agent.agent_id} ({agent.behavior_type}) sisteme eklendi!")

    def remove_agent(self, agent_id):
        """Sistemden bir ajanı çıkarır"""
        for i, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                self.behavioral_balance[agent.behavior_type] -= 1
                self.agents.pop(i)
                print(f"Agent {agent_id} sistemden çıkarıldı!")
                break

    def coordinate_learning(self):
        """Ajanlar arası koordineli öğrenme"""
        if not self.global_learning_enabled or len(self.agents) < 2:
            return
        
        current_step = sum(agent.total_training_steps for agent in self.agents)
        
        # Experience sharing periyodik olarak
        if current_step - self.last_sharing_step > self.experience_sharing_interval:
            self._facilitate_experience_sharing()
            self.last_sharing_step = current_step

    def _facilitate_experience_sharing(self):
        """Ajanlar arası deneyim paylaşımını kolaylaştırır"""
        for agent in self.agents:
            other_agents = [a for a in self.agents if a.agent_id != agent.agent_id]
            agent.share_experience(other_agents)
            
        print("🤝 Multi-agent experience sharing completed!")

    def update_competition_results(self, race_results):
        """Yarış sonuçlarını günceller ve analiz eder"""
        self.competition_results.append(race_results)
        
        # Winner tracking
        if 'winner' in race_results:
            self.race_winners.append(race_results['winner'])
        
        # Performance analysis
        performance_data = {}
        for agent in self.agents:
            performance_data[agent.agent_id] = {
                'behavior': agent.behavior_type,
                'difficulty': agent.difficulty,
                'stats': agent.get_learning_stats(),
                'position': race_results.get(f'agent_{agent.agent_id}_position', 'unknown')
            }
        
        self.performance_history.append(performance_data)
        
        # Adaptive difficulty ayarlaması
        self._adjust_adaptive_difficulties()

    def _adjust_adaptive_difficulties(self):
        """Performansa göre zorluk seviyelerini ayarlar"""
        if len(self.race_winners) < 10:  # En az 10 yarış gerekli
            return
        
        recent_winners = self.race_winners[-10:]
        winner_performance = {}
        
        # Son 10 yarışın kazananlarını analiz et
        for winner_id in recent_winners:
            if winner_id not in winner_performance:
                winner_performance[winner_id] = 0
            winner_performance[winner_id] += 1
        
        # Çok başarılı ajanların zorluğunu artır
        for agent in self.agents:
            agent_wins = winner_performance.get(agent.agent_id, 0)
            win_rate = agent_wins / 10
            
            performance_metrics = {
                'avg_reward': agent.stats['average_reward_last_100'],
                'win_rate': win_rate
            }
            
            agent.adaptive_difficulty_adjustment(performance_metrics)

    def get_system_statistics(self):
        """Sistem geneli istatistikleri döndürür"""
        if not self.agents:
            return {"error": "No agents in system"}
        
        total_episodes = sum(agent.stats['episodes_completed'] for agent in self.agents)
        total_laps = sum(agent.stats['laps_completed'] for agent in self.agents)
        avg_system_reward = sum(agent.stats['average_reward_last_100'] for agent in self.agents) / len(self.agents)
        
        behavioral_performance = {}
        for behavior in self.behavioral_balance.keys():
            agents_of_type = [a for a in self.agents if a.behavior_type == behavior]
            if agents_of_type:
                avg_performance = sum(a.stats['average_reward_last_100'] for a in agents_of_type) / len(agents_of_type)
                behavioral_performance[behavior] = avg_performance
        
        return {
            'total_agents': len(self.agents),
            'total_episodes': total_episodes,
            'total_laps': total_laps,
            'average_system_reward': avg_system_reward,
            'behavioral_distribution': self.behavioral_balance.copy(),
            'behavioral_performance': behavioral_performance,
            'total_races': len(self.competition_results),
            'recent_winners': self.race_winners[-5:] if len(self.race_winners) >= 5 else self.race_winners,
            'training_session': self.training_session_id
        }

    def balance_behavioral_diversity(self):
        """Davranışsal çeşitliliği dengelemeye çalışır"""
        total_agents = len(self.agents)
        if total_agents == 0:
            return
        
        # İdeal dağılım: Her davranış tipinden eşit sayıda
        ideal_per_type = total_agents // 4
        
        imbalanced_types = []
        for behavior, count in self.behavioral_balance.items():
            if abs(count - ideal_per_type) > 1:
                imbalanced_types.append((behavior, count, ideal_per_type))
        
        if imbalanced_types:
            print("⚖️ Behavioral imbalance detected:")
            for behavior, current, ideal in imbalanced_types:
                print(f"  {behavior}: {current} (ideal: {ideal})")

    def start_training_session(self):
        """Yeni bir training session başlatır"""
        self.training_session_id += 1
        print(f"🚀 Multi-agent training session #{self.training_session_id} started!")
        
        # Reset some statistics
        for agent in self.agents:
            agent.episode_steps = 0
        
        # Print session info
        stats = self.get_system_statistics()
        print(f"📊 System: {stats['total_agents']} agents, Avg Reward: {stats['average_system_reward']:.2f}")
