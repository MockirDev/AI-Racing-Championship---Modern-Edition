import pygame
import json
import time
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque, defaultdict
import numpy as np

class TelemetrySystem:
    """
    Comprehensive telemetry and statistics system for AI racing
    """
    def __init__(self):
        # Data Storage
        self.race_sessions = []
        self.current_session = None
        self.ai_performance_data = defaultdict(lambda: {
            'lap_times': [],
            'positions': [],
            'rewards': [],
            'collisions': 0,
            'boost_usage': 0,
            'distance_traveled': 0,
            'learning_progress': []
        })
        
        # Real-time tracking
        self.live_data_buffer = deque(maxlen=1000)  # Last 1000 frames
        self.session_start_time = 0
        self.frame_count = 0
        
        # Statistics
        self.global_stats = {
            'total_races': 0,
            'total_laps': 0,
            'fastest_lap_time': float('inf'),
            'fastest_lap_holder': None,
            'total_training_time': 0,
            'ai_win_rates': defaultdict(float),
            'track_records': defaultdict(lambda: {'time': float('inf'), 'holder': None})
        }
        
        # Visualization settings
        pygame.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Colors for different data types
        self.colors = {
            'speed': (255, 100, 100),
            'reward': (100, 255, 100), 
            'position': (100, 100, 255),
            'boost': (255, 255, 100),
            'learning': (255, 100, 255)
        }
        
        # Load existing data
        self.load_telemetry_data()

    def start_session(self, session_name, track_name, ai_agents):
        """Start a new telemetry session"""
        self.session_start_time = time.time()
        self.frame_count = 0
        
        self.current_session = {
            'session_name': session_name,
            'track_name': track_name,
            'start_time': datetime.now().isoformat(),
            'agents': [agent.name for agent in ai_agents],
            'agent_data': {agent.name: {
                'behavior_type': agent.behavior_type,
                'difficulty': agent.difficulty,
                'start_level': agent.customization['progression']['level'],
                'start_xp': agent.customization['progression']['xp']
            } for agent in ai_agents},
            'frame_data': [],
            'events': []
        }
        
        print(f"📊 Telemetry session started: {session_name}")

    def record_frame(self, cars, checkpoints, boost_zones=None):
        """Record telemetry data for current frame"""
        if not self.current_session:
            return
            
        self.frame_count += 1
        frame_time = time.time() - self.session_start_time
        
        frame_data = {
            'frame': self.frame_count,
            'time': frame_time,
            'cars': {}
        }
        
        for car in cars:
            if hasattr(car, 'ai_agent'):
                agent_stats = car.ai_agent.get_learning_stats()
            else:
                agent_stats = {'exploration': 0, 'memory_usage': 0}
                
            car_data = {
                'position': [car.position.x, car.position.y],
                'velocity': car.velocity.length(),
                'angle': car.angle,
                'alive': car.alive,
                'current_checkpoint': car.next_checkpoint_id,
                'laps_completed': car.laps_completed,
                'reward': getattr(car, 'reward', 0),
                'boost_active': getattr(car, 'boost_active', False),
                'level': car.customization['progression']['level'],
                'xp': car.customization['progression']['xp'],
                'sensors': getattr(car, 'sensors', []),
                'exploration_rate': agent_stats.get('exploration', 0),
                'memory_usage': agent_stats.get('memory_usage', 0)
            }
            
            frame_data['cars'][car.name] = car_data
        
        self.current_session['frame_data'].append(frame_data)
        
        # Add to live buffer for real-time display
        self.live_data_buffer.append(frame_data)

    def record_event(self, event_type, agent_name, details=None):
        """Record significant events during racing"""
        if not self.current_session:
            return
            
        event = {
            'frame': self.frame_count,
            'time': time.time() - self.session_start_time,
            'type': event_type,
            'agent': agent_name,
            'details': details or {}
        }
        
        self.current_session['events'].append(event)
        
        # Update performance data
        if event_type == 'lap_completed':
            lap_time = details.get('lap_time', 0)
            self.ai_performance_data[agent_name]['lap_times'].append(lap_time)
            
            if lap_time < self.global_stats['fastest_lap_time']:
                self.global_stats['fastest_lap_time'] = lap_time
                self.global_stats['fastest_lap_holder'] = agent_name
                
        elif event_type == 'collision':
            self.ai_performance_data[agent_name]['collisions'] += 1
            
        elif event_type == 'boost_used':
            self.ai_performance_data[agent_name]['boost_usage'] += 1
            
        elif event_type == 'level_up':
            level = details.get('new_level', 0)
            print(f"📈 {agent_name} reached level {level}!")

    def end_session(self):
        """End current telemetry session and save data"""
        if not self.current_session:
            return
            
        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['duration'] = time.time() - self.session_start_time
        self.current_session['total_frames'] = self.frame_count
        
        # Calculate session statistics
        self.calculate_session_stats()
        
        # Save session data
        self.race_sessions.append(self.current_session)
        self.save_telemetry_data()
        
        session_name = self.current_session['session_name']
        duration = self.current_session['duration']
        
        print(f"📊 Telemetry session ended: {session_name}")
        print(f"⏱️ Duration: {duration:.1f}s, Frames: {self.frame_count}")
        
        self.current_session = None

    def calculate_session_stats(self):
        """Calculate comprehensive statistics for the session"""
        if not self.current_session:
            return
            
        session_stats = {
            'average_speed': {},
            'completion_rates': {},
            'learning_progress': {},
            'performance_ranking': []
        }
        
        # Analyze each agent's performance
        for agent_name in self.current_session['agents']:
            agent_frames = []
            for frame in self.current_session['frame_data']:
                if agent_name in frame['cars']:
                    agent_frames.append(frame['cars'][agent_name])
            
            if agent_frames:
                # Average speed
                speeds = [frame['velocity'] for frame in agent_frames if frame['alive']]
                session_stats['average_speed'][agent_name] = np.mean(speeds) if speeds else 0
                
                # Completion rate
                max_checkpoint = max([frame['current_checkpoint'] for frame in agent_frames])
                total_checkpoints = len(self.current_session.get('checkpoints', [1]))  # Default to 1 if unknown
                session_stats['completion_rates'][agent_name] = max_checkpoint / max(total_checkpoints, 1)
                
                # Learning progress
                if agent_frames:
                    start_level = agent_frames[0]['level']
                    end_level = agent_frames[-1]['level']
                    session_stats['learning_progress'][agent_name] = end_level - start_level
        
        # Performance ranking
        for agent_name in self.current_session['agents']:
            laps = self.ai_performance_data[agent_name]['laps_completed']
            avg_speed = session_stats['average_speed'].get(agent_name, 0)
            collisions = self.ai_performance_data[agent_name]['collisions']
            
            score = (laps * 1000) + (avg_speed * 10) - (collisions * 100)
            session_stats['performance_ranking'].append({
                'agent': agent_name,
                'score': score,
                'laps': laps,
                'avg_speed': avg_speed,
                'collisions': collisions
            })
        
        # Sort by performance score
        session_stats['performance_ranking'].sort(key=lambda x: x['score'], reverse=True)
        
        self.current_session['statistics'] = session_stats

    def get_live_telemetry_display(self, screen, x, y, width, height):
        """Create real-time telemetry display overlay"""
        if not self.live_data_buffer:
            return
            
        # Create telemetry surface
        telemetry_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        telemetry_surface.fill((0, 0, 0, 180))
        
        # Title
        title = self.font.render("📊 Live Telemetry", True, (255, 255, 255))
        telemetry_surface.blit(title, (10, 10))
        
        y_offset = 40
        
        # Get latest frame data
        if self.live_data_buffer:
            latest_frame = self.live_data_buffer[-1]
            
            # Display current stats for each car
            for car_name, car_data in latest_frame['cars'].items():
                if car_data['alive']:
                    # Car name and level
                    name_text = f"{car_name} (Lvl {car_data['level']})"
                    name_surface = self.small_font.render(name_text, True, (255, 255, 100))
                    telemetry_surface.blit(name_surface, (15, y_offset))
                    y_offset += 20
                    
                    # Speed
                    speed_text = f"  Speed: {car_data['velocity']:.1f}"
                    speed_surface = self.small_font.render(speed_text, True, self.colors['speed'])
                    telemetry_surface.blit(speed_surface, (20, y_offset))
                    y_offset += 15
                    
                    # Progress (checkpoint/lap info)
                    progress_text = f"  Progress: CP{car_data['current_checkpoint']} L{car_data['laps_completed']}"
                    progress_surface = self.small_font.render(progress_text, True, (200, 200, 200))
                    telemetry_surface.blit(progress_surface, (20, y_offset))
                    y_offset += 15
                    
                    # XP and rewards
                    if car_data['reward'] != 0:
                        reward_text = f"  Reward: {car_data['reward']:.2f}"
                        reward_surface = self.small_font.render(reward_text, True, self.colors['reward'])
                        telemetry_surface.blit(reward_surface, (20, y_offset))
                        y_offset += 15
                    
                    # Boost status
                    if car_data['boost_active']:
                        boost_text = "  🚀 BOOST ACTIVE"
                        boost_surface = self.small_font.render(boost_text, True, self.colors['boost'])
                        telemetry_surface.blit(boost_surface, (20, y_offset))
                        y_offset += 15
                    
                    y_offset += 5  # Space between cars
        
        # Session info
        if self.current_session:
            session_time = time.time() - self.session_start_time
            session_info = f"Session: {session_time:.1f}s | Frame: {self.frame_count}"
            session_surface = self.small_font.render(session_info, True, (150, 150, 150))
            telemetry_surface.blit(session_surface, (10, height - 25))
        
        screen.blit(telemetry_surface, (x, y))

    def export_session_data(self, session_index, format='csv'):
        """Export session data to CSV or JSON"""
        if session_index >= len(self.race_sessions):
            return False
            
        session = self.race_sessions[session_index]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            filename = f"telemetry_{session['session_name']}_{timestamp}.csv"
            return self._export_to_csv(session, filename)
        elif format == 'json':
            filename = f"telemetry_{session['session_name']}_{timestamp}.json"
            return self._export_to_json(session, filename)
        
        return False

    def _export_to_csv(self, session, filename):
        """Export session data to CSV format"""
        try:
            os.makedirs('telemetry_exports', exist_ok=True)
            filepath = os.path.join('telemetry_exports', filename)
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow(['Frame', 'Time', 'Agent', 'Position_X', 'Position_Y', 
                               'Velocity', 'Angle', 'Alive', 'Checkpoint', 'Laps', 
                               'Reward', 'Boost', 'Level', 'XP'])
                
                # Data rows
                for frame in session['frame_data']:
                    for agent_name, car_data in frame['cars'].items():
                        writer.writerow([
                            frame['frame'], frame['time'], agent_name,
                            car_data['position'][0], car_data['position'][1],
                            car_data['velocity'], car_data['angle'],
                            car_data['alive'], car_data['current_checkpoint'],
                            car_data['laps_completed'], car_data['reward'],
                            car_data['boost_active'], car_data['level'], car_data['xp']
                        ])
            
            print(f"📁 Telemetry data exported to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

    def _export_to_json(self, session, filename):
        """Export session data to JSON format"""
        try:
            os.makedirs('telemetry_exports', exist_ok=True)
            filepath = os.path.join('telemetry_exports', filename)
            
            with open(filepath, 'w') as jsonfile:
                json.dump(session, jsonfile, indent=2)
            
            print(f"📁 Telemetry data exported to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.race_sessions:
            return "No telemetry data available."
        
        report = []
        report.append("=" * 50)
        report.append("🏁 AI RACING PERFORMANCE REPORT")
        report.append("=" * 50)
        
        # Global statistics
        report.append(f"📊 Total Races: {self.global_stats['total_races']}")
        report.append(f"🏃 Total Laps: {self.global_stats['total_laps']}")
        
        if self.global_stats['fastest_lap_holder']:
            report.append(f"⚡ Fastest Lap: {self.global_stats['fastest_lap_time']:.2f}s by {self.global_stats['fastest_lap_holder']}")
        
        # Recent session analysis
        if self.race_sessions:
            latest_session = self.race_sessions[-1]
            report.append(f"\n🔍 Latest Session: {latest_session['session_name']}")
            report.append(f"📅 Date: {latest_session['start_time'][:19]}")
            report.append(f"⏱️ Duration: {latest_session.get('duration', 0):.1f}s")
            
            if 'statistics' in latest_session:
                stats = latest_session['statistics']
                
                # Performance ranking
                report.append(f"\n🏆 Performance Ranking:")
                for i, ranking in enumerate(stats['performance_ranking'][:5]):  # Top 5
                    report.append(f"  {i+1}. {ranking['agent']} - Score: {ranking['score']:.0f}")
                
                # Learning progress
                report.append(f"\n📈 Learning Progress:")
                for agent, progress in stats['learning_progress'].items():
                    report.append(f"  {agent}: +{progress} levels")
        
        return "\n".join(report)

    def save_telemetry_data(self):
        """Save telemetry data to file"""
        try:
            os.makedirs('telemetry_data', exist_ok=True)
            
            # Save sessions data
            with open('telemetry_data/sessions.json', 'w') as f:
                json.dump(self.race_sessions, f, indent=2)
            
            # Save global stats
            with open('telemetry_data/global_stats.json', 'w') as f:
                json.dump(self.global_stats, f, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to save telemetry data: {e}")

    def load_telemetry_data(self):
        """Load existing telemetry data"""
        try:
            # Load sessions
            if os.path.exists('telemetry_data/sessions.json'):
                with open('telemetry_data/sessions.json', 'r') as f:
                    self.race_sessions = json.load(f)
            
            # Load global stats  
            if os.path.exists('telemetry_data/global_stats.json'):
                with open('telemetry_data/global_stats.json', 'r') as f:
                    loaded_stats = json.load(f)
                    self.global_stats.update(loaded_stats)
                    
        except Exception as e:
            print(f"⚠️ Could not load telemetry data: {e}")

# Global telemetry instance
telemetry = TelemetrySystem()
