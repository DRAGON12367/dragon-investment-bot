"""
AI Investment Bot - Main Entry Point
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from algorithms.strategy import TradingStrategy
from algorithms.profit_analyzer import ProfitAnalyzer
from algorithms.aggressive_growth import AggressiveGrowthStrategy
from algorithms.investment_advisor import InvestmentAdvisor
from risk_management.risk_manager import RiskManager
from utils.config import Config
from utils.logger import setup_logger
from web_automation.broker_client import BrokerClient
from data.data_processor import DataProcessor


class InvestmentBot:
    """Main investment bot class that orchestrates trading operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the investment bot with configuration."""
        self.config = Config(config_path)
        self.logger = setup_logger(self.config.log_level, self.config.log_file)
        self.strategy = TradingStrategy(self.config)
        self.profit_analyzer = ProfitAnalyzer(self.config)
        self.aggressive_growth = AggressiveGrowthStrategy(self.config)
        self.investment_advisor = InvestmentAdvisor(self.config)
        self.risk_manager = RiskManager(self.config)
        self.broker_client = BrokerClient(self.config)
        self.data_processor = DataProcessor(self.config)
        
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing AI Investment Bot...")
        await self.broker_client.connect()
        await self.strategy.initialize()
        self.logger.info("Bot initialized successfully")
        
    async def run(self):
        """Main execution loop."""
        try:
            await self.initialize()
            
            while True:
                try:
                    # Scan both stock and crypto markets
                    self.logger.info("Scanning markets (stocks & crypto)...")
                    market_data = await self.broker_client.get_all_market_data()
                    
                    if not market_data:
                        self.logger.warning("No market data received, skipping iteration")
                        await asyncio.sleep(self.config.trading_interval)
                        continue
                    
                    # Log what we scanned
                    stocks = [s for s, d in market_data.items() if d.get("asset_type") == "stock"]
                    crypto = [s for s, d in market_data.items() if d.get("asset_type") == "crypto"]
                    self.logger.info(f"Scanned {len(stocks)} stocks and {len(crypto)} cryptocurrencies")
                    
                    # Save market data for historical analysis
                    self.data_processor.save_market_data(market_data)
                    
                    # Get portfolio status for aggressive growth analysis
                    portfolio_status = await self.broker_client.get_portfolio_status()
                    portfolio_value = portfolio_status.get('total_value', 100.0)
                    current_positions = portfolio_status.get('positions', [])
                    
                    # 💡 INVESTMENT ADVISOR - Live 24/7 Guidance for Small Accounts
                    # Only provide guidance if we have market data
                    if market_data:
                        try:
                            investment_guidance = self.investment_advisor.get_investment_guidance(
                                portfolio_value,
                                market_data,
                                current_positions,
                                historical_data if historical_data else None
                            )
                        except Exception as e:
                            self.logger.error(f"Error generating investment guidance: {e}")
                            investment_guidance = {'status': 'error', 'message': str(e)}
                    else:
                        investment_guidance = {
                            'status': 'no_data',
                            'message': 'No market data available. Waiting for data...'
                        }
                    
                    # Display investment guidance
                    if investment_guidance.get('status') != 'insufficient_funds':
                        self.logger.info("=" * 80)
                        self.logger.info("💡 LIVE INVESTMENT GUIDANCE - What to Invest In Right Now")
                        self.logger.info("=" * 80)
                        self.logger.info(f"Account: ${portfolio_value:.2f} ({investment_guidance.get('account_size_category', 'UNKNOWN')})")
                        
                        recommendations = investment_guidance.get('recommendations', [])
                        if recommendations:
                            # Separate crypto and stock recommendations
                            crypto_recs = [r for r in recommendations if r.get('asset_type') == 'crypto']
                            stock_recs = [r for r in recommendations if r.get('asset_type') == 'stock']
                            
                            # Show crypto recommendations first (prioritize showing many cryptos)
                            if crypto_recs:
                                self.logger.info(f"\n🚀 TOP {len(crypto_recs)} CRYPTO RECOMMENDATIONS:")
                                for i, rec in enumerate(crypto_recs, 1):
                                    self.logger.info(
                                        f"  {i}. {rec['symbol']} (CRYPTO) - {rec['action']}\n"
                                        f"     💰 Invest: ${rec['position_value']:.2f} ({rec['position_size_pct']:.1%} of portfolio)\n"
                                        f"     📊 Entry: ${rec['current_price']:.2f} | Target: ${rec['target_price']:.2f} (+{rec['expected_return_pct']:.1f}%)\n"
                                        f"     🛡️ Stop Loss: ${rec['stop_loss']:.2f} | Risk: {rec['risk_level']}\n"
                                        f"     ⏱️ Timeframe: {rec['timeframe']} | Confidence: {rec['confidence']:.1%}\n"
                                        f"     💡 Reason: {rec['reason']}"
                                    )
                            
                            # Show stock recommendations
                            if stock_recs:
                                self.logger.info(f"\n📈 TOP {len(stock_recs)} STOCK RECOMMENDATIONS:")
                                for i, rec in enumerate(stock_recs, 1):
                                    self.logger.info(
                                        f"  {i}. {rec['symbol']} (STOCK) - {rec['action']}\n"
                                        f"     💰 Invest: ${rec['position_value']:.2f} ({rec['position_size_pct']:.1%} of portfolio)\n"
                                        f"     📊 Entry: ${rec['current_price']:.2f} | Target: ${rec['target_price']:.2f} (+{rec['expected_return_pct']:.1f}%)\n"
                                        f"     🛡️ Stop Loss: ${rec['stop_loss']:.2f} | Risk: {rec['risk_level']}\n"
                                        f"     ⏱️ Timeframe: {rec['timeframe']} | Confidence: {rec['confidence']:.1%}\n"
                                        f"     💡 Reason: {rec['reason']}"
                                    )
                            
                            # Summary
                            self.logger.info(f"\n📊 SUMMARY: {len(crypto_recs)} Crypto + {len(stock_recs)} Stock = {len(recommendations)} Total Recommendations")
                        else:
                            self.logger.info("\n⚠️ No high-conviction opportunities found at this time. Waiting for better setups...")
                        
                        # Show growth projection
                        growth_proj = investment_guidance.get('growth_projection', {})
                        if growth_proj and 'projections' in growth_proj:
                            self.logger.info(f"\n📈 GROWTH PROJECTION:")
                            proj = growth_proj['projections']
                            for timeframe, data in proj.items():
                                months = timeframe.replace('_month', '')
                                self.logger.info(
                                    f"  {months} month(s): ${data['projected_value']:.2f} "
                                    f"(+{data['gain_percentage']:.1f}% / +${data['potential_gain']:.2f})"
                                )
                            
                            years_to_100k = growth_proj.get('target_100k_years')
                            if years_to_100k:
                                self.logger.info(f"  🎯 Estimated time to $100,000: {years_to_100k:.1f} years")
                        
                        # Show action plan
                        action_plan = investment_guidance.get('action_plan', {})
                        if action_plan.get('immediate_actions'):
                            self.logger.info(f"\n🚀 ACTION PLAN:")
                            for action in action_plan['immediate_actions']:
                                self.logger.info(
                                    f"  {action['message']}\n"
                                    f"     Symbols: {', '.join(action['symbols'])}\n"
                                    f"     Investment: ${action['total_investment']:.2f}"
                                )
                        
                        # Risk assessment
                        risk = investment_guidance.get('risk_assessment', {})
                        if risk:
                            self.logger.info(f"\n⚠️ RISK ASSESSMENT:")
                            self.logger.info(f"  Overall Risk: {risk.get('overall_risk_level', 'UNKNOWN')}")
                            self.logger.info(f"  Diversification: {risk.get('diversification', 'UNKNOWN')}")
                            self.logger.info(f"  Recommendation: {risk.get('recommendation', 'N/A')}")
                        
                        self.logger.info("=" * 80)
                    else:
                        self.logger.warning(f"⚠️ {investment_guidance.get('message', 'Insufficient funds')}")
                    
                    # 🚀 AGGRESSIVE GROWTH STRATEGY - Turn $100 into $100,000
                    # Optionally load historical data for better analysis
                    historical_data = {}
                    for symbol in list(market_data.keys())[:20]:  # Limit to first 20 for performance
                        hist_df = self.data_processor.load_historical_data(symbol, days=50)
                        if not hist_df.empty:
                            historical_data[symbol] = hist_df
                    
                    growth_opportunities = self.aggressive_growth.analyze_growth_opportunities(
                        market_data,
                        portfolio_value,
                        historical_data if historical_data else None
                    )
                    
                    if growth_opportunities:
                        self.logger.info(f"🚀 AGGRESSIVE GROWTH: Found {len(growth_opportunities)} high-conviction opportunities!")
                        for i, opp in enumerate(growth_opportunities[:3], 1):
                            self.logger.info(
                                f"  {i}. {opp['symbol']} ({opp['asset_type']}): "
                                f"Conviction: {opp['conviction_score']:.1%} | "
                                f"Growth Potential: {opp['growth_potential']:.1%} | "
                                f"Position Size: {opp['position_size_pct']:.1%} | "
                                f"Target: ${opp['target_price']:.2f} (+{opp['expected_return']:.1%})"
                            )
                        
                        # Show compound growth plan
                        growth_plan = self.aggressive_growth.calculate_compound_growth_plan(
                            portfolio_value,
                            100000.0  # Target: $100,000
                        )
                        if growth_plan:
                            self.logger.info(
                                f"💰 Growth Plan: ${portfolio_value:.2f} → $100,000 "
                                f"({growth_plan['progress_pct']:.1f}% progress)"
                            )
                    
                    # AI Profit Analysis - Find most profitable opportunities
                    opportunities = self.profit_analyzer.analyze_opportunities(market_data)
                    top_opportunities = self.profit_analyzer.get_top_opportunities(opportunities, limit=10)
                    
                    if top_opportunities:
                        self.logger.info(f"🏆 Top {len(top_opportunities)} Profit Opportunities:")
                        for i, opp in enumerate(top_opportunities[:5], 1):
                            self.logger.info(
                                f"  {i}. {opp['symbol']} ({opp['asset_type']}): "
                                f"{opp['action']} @ ${opp['current_price']:.2f} - "
                                f"Profit Score: {opp['profit_score']:.2%} | "
                                f"Target: ${opp['target_price']:.2f} (+{opp['profit_potential']:.1%})"
                            )
                    
                    # Check for sell signals on existing positions
                    sell_signals = self.profit_analyzer.analyze_sell_signals(market_data)
                    if sell_signals:
                        self.logger.warning(f"⚠️ {len(sell_signals)} SELL SIGNALS DETECTED:")
                        for signal in sell_signals:
                            self.logger.warning(
                                f"  🚨 SELL {signal['symbol']}: {signal['message']} | "
                                f"Profit: {signal['profit_percent']:.2f}% | "
                                f"Priority: {signal['priority']}"
                            )
                    
                    # Generate trading signals using ML and technical analysis with Wall Street features
                    signals = await self.strategy.generate_signals(
                        market_data,
                        historical_data if historical_data else None
                    )
                    
                    # Add aggressive growth opportunities as highest priority signals
                    for opp in growth_opportunities[:3]:  # Top 3 aggressive growth opportunities
                        signals.append({
                            'symbol': opp['symbol'],
                            'asset_type': opp['asset_type'],
                            'action': 'BUY',
                            'confidence': opp['conviction_score'],
                            'price': opp['current_price'],
                            'target_price': opp['target_price'],
                            'stop_loss': opp['stop_loss'],
                            'take_profit': opp['target_price'],
                            'profit_potential': opp['growth_potential'],
                            'risk_reward_ratio': opp['risk_reward_ratio'],
                            'position_size_pct': opp['position_size_pct'],
                            'position_value': opp['position_value'],
                            'timestamp': datetime.now().isoformat(),
                            'source': 'AGGRESSIVE_GROWTH'
                        })
                    
                    # Add top opportunities as strong buy signals
                    for opp in top_opportunities[:5]:  # Top 5 opportunities
                        if opp['action'] in ['STRONG_BUY', 'BUY'] and opp['profit_score'] > 0.5:
                            signals.append({
                                'symbol': opp['symbol'],
                                'asset_type': opp['asset_type'],
                                'action': 'BUY',
                                'confidence': opp['profit_score'],
                                'price': opp['current_price'],
                                'target_price': opp['target_price'],
                                'stop_loss': opp['stop_loss'],
                                'take_profit': opp['target_price'],
                                'profit_potential': opp['profit_potential'],
                                'risk_reward_ratio': opp['risk_reward_ratio'],
                                'timestamp': datetime.now().isoformat(),
                                'source': 'AI_PROFIT_ANALYZER'
                            })
                    
                    if signals:
                        stock_signals = [s for s in signals if s.get("asset_type") == "stock"]
                        crypto_signals = [s for s in signals if s.get("asset_type") == "crypto"]
                        self.logger.info(
                            f"Generated {len(signals)} trading signals "
                            f"({len(stock_signals)} stocks, {len(crypto_signals)} crypto)"
                        )
                    
                    # Apply risk management (but allow aggressive growth signals through)
                    approved_signals = self.risk_manager.evaluate_signals(
                        signals, 
                        portfolio_status
                    )
                    
                    # Prioritize aggressive growth signals (they have higher position sizes)
                    aggressive_signals = [s for s in approved_signals if s.get('source') == 'AGGRESSIVE_GROWTH']
                    other_signals = [s for s in approved_signals if s.get('source') != 'AGGRESSIVE_GROWTH']
                    
                    # Reorder: aggressive growth first, then others
                    approved_signals = aggressive_signals + other_signals
                    
                    # Execute sell signals first (priority)
                    if sell_signals:
                        sell_orders = [{
                            'symbol': s['symbol'],
                            'action': 'SELL',
                            'price': s['current_price'],
                            'quantity': 0,  # Will be set by risk manager
                            'reason': s['reason'],
                            'priority': s['priority']
                        } for s in sell_signals]
                        
                        # Get current positions to determine quantities
                        for order in sell_orders:
                            for pos in portfolio_status.get('positions', []):
                                if pos.get('symbol') == order['symbol']:
                                    order['quantity'] = pos.get('quantity', 0)
                                    break
                        
                        sell_orders = [o for o in sell_orders if o['quantity'] > 0]
                        if sell_orders:
                            self.logger.warning(f"🚨 Executing {len(sell_orders)} SELL orders")
                            results = await self.broker_client.execute_trades(sell_orders)
                            for result in results:
                                if "error" not in result:
                                    # Remove from tracking
                                    symbol = result.get('symbol')
                                    if symbol in self.profit_analyzer.tracked_positions:
                                        del self.profit_analyzer.tracked_positions[symbol]
                                    self.logger.warning(f"✅ SELL executed: {symbol} @ ${result.get('price', 0):.2f}")
                    
                    # Execute buy trades (simulated/paper trading)
                    buy_signals = [s for s in approved_signals if s.get('action') == 'BUY']
                    if buy_signals:
                        stock_trades = [s for s in buy_signals if s.get("asset_type") == "stock"]
                        crypto_trades = [s for s in buy_signals if s.get("asset_type") == "crypto"]
                        self.logger.info(
                            f"Executing {len(buy_signals)} approved BUY trades "
                            f"({len(stock_trades)} stocks, {len(crypto_trades)} crypto)"
                        )
                        results = await self.broker_client.execute_trades(buy_signals)
                        for result in results:
                            if "error" not in result:
                                symbol = result.get('symbol')
                                price = result.get('price', 0)
                                quantity = result.get('quantity', 0)
                                
                                # Start tracking this position
                                asset_type = next((s.get('asset_type') for s in buy_signals if s.get('symbol') == symbol), 'unknown')
                                self.profit_analyzer.track_position(symbol, price, quantity, asset_type)
                                
                                # Track aggressive growth trades
                                signal_source = next((s.get('source') for s in buy_signals if s.get('symbol') == symbol), '')
                                if signal_source == 'AGGRESSIVE_GROWTH':
                                    self.aggressive_growth.track_performance({
                                        'symbol': symbol,
                                        'entry_price': price,
                                        'quantity': quantity,
                                        'position_value': price * quantity,
                                        'source': 'AGGRESSIVE_GROWTH'
                                    })
                                
                                self.logger.info(f"✅ BUY executed: {symbol} {quantity} @ ${price:.2f} (Source: {signal_source})")
                    else:
                        self.logger.debug("No approved buy signals to execute")
                    
                    # Log portfolio status periodically
                    positions = portfolio_status.get("positions", [])
                    self.logger.info(
                        f"Portfolio: ${portfolio_status.get('total_value', 0):.2f} total, "
                        f"${portfolio_status.get('cash', 0):.2f} cash, "
                        f"{len(positions)} positions"
                    )
                    
                    # Show aggressive growth performance summary
                    growth_performance = self.aggressive_growth.get_performance_summary()
                    if growth_performance.get('total_trades', 0) > 0:
                        self.logger.info(
                            f"🚀 Aggressive Growth Performance: "
                            f"{growth_performance['win_rate']:.1%} win rate | "
                            f"Avg Win: {growth_performance['avg_win_pct']:.1f}% | "
                            f"Compound Growth Rate: {growth_performance['compound_growth_rate']:.1f}%"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}", exc_info=True)
                    await asyncio.sleep(5)  # Wait before retrying
                
                # Wait for next iteration
                await asyncio.sleep(self.config.trading_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except asyncio.CancelledError:
            self.logger.info("Bot cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            try:
                await self.cleanup()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
            
    async def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up...")
        await self.broker_client.disconnect()


async def main():
    """Main entry point."""
    bot = InvestmentBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())

