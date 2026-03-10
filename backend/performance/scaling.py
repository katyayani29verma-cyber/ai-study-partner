"""Scaling and load balancing strategies"""
import logging
from typing import List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"


class Server:
    """Server instance"""
    
    def __init__(self, host: str, port: int, weight: int = 1):
        self.host = host
        self.port = port
        self.weight = weight
        self.active_connections = 0
        self.total_requests = 0
        self.is_healthy = True
    
    def get_url(self) -> str:
        """Get server URL"""
        return f"http://{self.host}:{self.port}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "is_healthy": self.is_healthy,
        }


class LoadBalancer:
    """Load balancer for distributing traffic"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.servers: List[Server] = []
        self.strategy = strategy
        self.current_index = 0
    
    def add_server(self, host: str, port: int, weight: int = 1) -> None:
        """Add server to pool"""
        server = Server(host, port, weight)
        self.servers.append(server)
        logger.info(f"Server added: {server.get_url()}")
    
    def remove_server(self, host: str, port: int) -> None:
        """Remove server from pool"""
        self.servers = [s for s in self.servers if not (s.host == host and s.port == port)]
        logger.info(f"Server removed: {host}:{port}")
    
    def get_healthy_servers(self) -> List[Server]:
        """Get healthy servers"""
        return [s for s in self.servers if s.is_healthy]
    
    def select_server(self, client_ip: str = None) -> Server:
        """Select server based on strategy"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1
            return server
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_servers, key=lambda s: s.active_connections)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            total_weight = sum(s.weight for s in healthy_servers)
            weighted_index = (self.current_index % total_weight)
            self.current_index += 1
            
            cumulative = 0
            for server in healthy_servers:
                cumulative += server.weight
                if weighted_index < cumulative:
                    return server
            
            return healthy_servers[0]
        
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            if client_ip:
                hash_value = hash(client_ip) % len(healthy_servers)
                return healthy_servers[hash_value]
            return healthy_servers[0]
        
        return healthy_servers[0]
    
    def mark_server_healthy(self, host: str, port: int) -> None:
        """Mark server as healthy"""
        for server in self.servers:
            if server.host == host and server.port == port:
                server.is_healthy = True
                logger.info(f"Server marked healthy: {server.get_url()}")
    
    def mark_server_unhealthy(self, host: str, port: int) -> None:
        """Mark server as unhealthy"""
        for server in self.servers:
            if server.host == host and server.port == port:
                server.is_healthy = False
                logger.warning(f"Server marked unhealthy: {server.get_url()}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "strategy": self.strategy,
            "total_servers": len(self.servers),
            "healthy_servers": len(self.get_healthy_servers()),
            "servers": [s.to_dict() for s in self.servers],
        }


class AutoScaler:
    """Auto-scaling based on metrics"""
    
    def __init__(self, min_servers: int = 1, max_servers: int = 10):
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.current_servers = min_servers
        self.cpu_threshold = 80
        self.memory_threshold = 85
    
    def should_scale_up(self, cpu_percent: float, memory_percent: float) -> bool:
        """Check if should scale up"""
        return (cpu_percent > self.cpu_threshold or 
                memory_percent > self.memory_threshold) and \
               self.current_servers < self.max_servers
    
    def should_scale_down(self, cpu_percent: float, memory_percent: float) -> bool:
        """Check if should scale down"""
        return (cpu_percent < 30 and memory_percent < 40) and \
               self.current_servers > self.min_servers
    
    def scale_up(self) -> int:
        """Scale up by adding server"""
        if self.current_servers < self.max_servers:
            self.current_servers += 1
            logger.info(f"Scaled up to {self.current_servers} servers")
        return self.current_servers
    
    def scale_down(self) -> int:
        """Scale down by removing server"""
        if self.current_servers > self.min_servers:
            self.current_servers -= 1
            logger.info(f"Scaled down to {self.current_servers} servers")
        return self.current_servers
    
    def get_status(self) -> dict:
        """Get auto-scaler status"""
        return {
            "current_servers": self.current_servers,
            "min_servers": self.min_servers,
            "max_servers": self.max_servers,
            "cpu_threshold": self.cpu_threshold,
            "memory_threshold": self.memory_threshold,
        }


class ConnectionPool:
    """Database connection pool"""
    
    def __init__(self, pool_size: int = 10, max_overflow: int = 20):
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.available_connections = pool_size
        self.active_connections = 0
    
    def acquire_connection(self) -> bool:
        """Acquire connection from pool"""
        if self.available_connections > 0:
            self.available_connections -= 1
            self.active_connections += 1
            return True
        elif self.active_connections < (self.pool_size + self.max_overflow):
            self.active_connections += 1
            return True
        return False
    
    def release_connection(self) -> None:
        """Release connection back to pool"""
        if self.active_connections > 0:
            self.active_connections -= 1
            if self.available_connections < self.pool_size:
                self.available_connections += 1
    
    def get_stats(self) -> dict:
        """Get connection pool statistics"""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "available_connections": self.available_connections,
            "active_connections": self.active_connections,
            "utilization": (self.active_connections / 
                          (self.pool_size + self.max_overflow) * 100),
        }
