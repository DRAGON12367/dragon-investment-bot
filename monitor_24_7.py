"""
24/7 AI Monitoring Service - Continuously finds most profitable opportunities
"""
import asyncio
import logging
from datetime import datetime
from algorithms.profit_analyzer import ProfitAnalyzer
from web_automation.market_scanner import MarketScanner
from algorithms.strategy import TradingStrategy
from risk_management.risk_manager import RiskManager
from web_automation.broker_client import BrokerClient
from utils.config import Config
from utils.logger import setup_logger


class Monitor247:
    """24/7 AI monitoring service for stocks and crypto."""
    
    def __init__(self, config_path: str = None):
        """Initialize 24/7 monitor."""
        self.config = Config(config_path)
        self.logger = setup_logger(self.config.log_level, self.config.log_file)
        self.scanner = MarketScanner(self.config)
        self.strategy = TradingStrategy(self.config)
        self.profit_analyzer = ProfitAnalyzer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.broker_client = BrokerClient(self.config)
        
        # Monitoring stats
        self.scan_count = 0
        self.opportunities_found = 0
        self.sell_signals_triggered = 0
        
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("🚀 Initializing 24/7 AI Monitor...")
        await self.scanner.connect()
        await self.strategy.initialize()
        self.logger.info("✅ 24/7 Monitor initialized - Starting continuous scanning...")
    
    async def scan_and_analyze(self):
        """Perform one scan and analysis cycle."""
        try:
            self.scan_count += 1
            self.logger.info(f"📊 Scan #{self.scan_count} - Analyzing markets...")
            
            # Scan all markets
            market_data = await self.broker_client.get_all_market_data()
            
            if not market_data:
                self.logger.warning("⚠️ No market data received")
                return
            
            stocks = [s for s, d in market_data.items() if d.get("asset_type") == "stock"]
            crypto = [s for s, d in market_data.items() if d.get("asset_type") == "crypto"]
            
            self.logger.info(f"📈 Scanned {len(stocks)} stocks + {len(crypto)} cryptocurrencies")
            
            # AI Profit Analysis
            opportunities = self.profit_analyzer.analyze_opportunities(market_data)
            top_opportunities = self.profit_analyzer.get_top_opportunities(opportunities, limit=10)
            
            if top_opportunities:
                self.opportunities_found += len(top_opportunities)
                self.logger.info("🏆 TOP PROFIT OPPORTUNITIES:")
                for i, opp in enumerate(top_opportunities[:5], 1):
                    self.logger.info(
                        f"  {i}. {opp['symbol']} ({opp['asset_type']}) - "
                        f"{opp['action']} @ ${opp['current_price']:.2f} | "
                        f"Profit Score: {opp['profit_score']:.1%} | "
                        f"Target: ${opp['target_price']:.2f} (+{opp['profit_potential']:.1%})"
                    )
            
            # Check sell signals
            sell_signals = self.profit_analyzer.analyze_sell_signals(market_data)
            
            if sell_signals:
                self.sell_signals_triggered += len(sell_signals)
                self.logger.warning("🚨 SELL SIGNALS DETECTED:")
                for signal in sell_signals:
                    priority = signal.get('priority', 'MEDIUM')
                    if priority in ['CRITICAL', 'HIGH']:
                        self.logger.error(
                            f"  🚨 {signal['symbol']}: {signal['message']} | "
                            f"Profit: {signal['profit_percent']:.2f}% | "
                            f"SELL @ ${signal['current_price']:.2f}"
                        )
                    else:
                        self.logger.warning(
                            f"  ⚠️ {signal['symbol']}: {signal['message']} | "
                            f"Profit: {signal['profit_percent']:.2f}%"
                        )
            
            # Generate ML signals
            signals = await self.strategy.generate_signals(market_data)
            
            if signals:
                strong_signals = [s for s in signals if s.get('confidence', 0) > 0.8]
                self.logger.info(f"🎯 Generated {len(signals)} signals ({len(strong_signals)} high confidence)")
            
            # Get portfolio status
            portfolio = await self.broker_client.get_portfolio_status()
            self.logger.info(
                f"💰 Portfolio: ${portfolio.get('total_value', 0):,.2f} | "
                f"Positions: {len(portfolio.get('positions', []))} | "
                f"Cash: ${portfolio.get('cash', 0):,.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in scan cycle: {e}", exc_info=True)
    
    async def run_247(self):
        """Run 24/7 monitoring loop."""
        try:
            await self.initialize()
            
            self.logger.info("=" * 60)
            self.logger.info("🌙 24/7 AI MONITORING ACTIVE")
            self.logger.info("=" * 60)
            self.logger.info("Continuously scanning for profitable opportunities...")
            self.logger.info("Press Ctrl+C to stop")
            self.logger.info("=" * 60)
            
            while True:
                await self.scan_and_analyze()
                
                # Log summary every 10 scans
                if self.scan_count % 10 == 0:
                    self.logger.info("=" * 60)
                    self.logger.info(f"📊 Summary: {self.scan_count} scans | "
                                   f"{self.opportunities_found} opportunities | "
                                   f"{self.sell_signals_triggered} sell signals")
                    self.logger.info("=" * 60)
                
                # Wait for next scan
                await asyncio.sleep(self.config.trading_interval)
                
        except KeyboardInterrupt:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("🛑 24/7 Monitor stopped by user")
            self.logger.info(f"Final Stats: {self.scan_count} scans | "
                           f"{self.opportunities_found} opportunities found | "
                           f"{self.sell_signals_triggered} sell signals")
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"Fatal error in 24/7 monitor: {e}", exc_info=True)
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.scanner.disconnect()
        self.logger.info("✅ Cleanup complete")


async def main():
    """Main entry point for 24/7 monitoring."""
    monitor = Monitor247()
    await monitor.run_247()


if __name__ == "__main__":
    asyncio.run(main())

