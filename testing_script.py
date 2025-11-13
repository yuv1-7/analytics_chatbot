"""
Enhanced Automated Testing Script for Pharma Analytics Agent
Properly tracks ALL SQL queries including retries, feedback loops, and comprehensive state tracking

Usage:
    python test_automation_enhanced.py

Output:
    test_results/
        test_results_YYYYMMDD_HHMMSS.xlsx (main results)
        plots/*.html (visualization files)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from core.database import initialize_connection_pool, close_connection_pool
import traceback

load_dotenv()


class EnhancedAgentTester:
    """Enhanced testing with comprehensive SQL and state tracking"""
    
    def __init__(self, config_file: str = "test_queries.json"):
        """Initialize tester with result directory structure"""
        self.config_file = config_file
        self.test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.run_dir = self.results_dir / f"run_{self.test_run_id}"
        self.run_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        print(f"✓ Created test run directory: {self.run_dir}")
        print(f"✓ Plots directory: {self.plots_dir}")
        
        # Results storage
        self.test_results = []
        self.sql_queries = []  # Track ALL SQL queries
        self.execution_details = []  # Track detailed execution flow
    
    def load_test_queries(self) -> List[str]:
        """Load test queries from JSON config file"""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            print(f"Config file not found: {self.config_file}")
            print("Creating default test_queries.json...")
            self._create_default_config()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        queries = config.get('test_queries', [])
        print(f"Loaded {len(queries)} test queries from {self.config_file}")
        return queries
    
    def _create_default_config(self):
        """Create default test queries configuration file"""
        default_queries = {
            "test_queries": [
                "Compare Random Forest and XGBoost for NRx forecasting",
                "Compare Random Forest and XGBoost for NRx forecasting",
                "Compare Random Forest and XGBoost for NRx forecasting",
                "Show me model drift for the ensemble models for hcp forecasting",
                "Show me model drift for the ensemble models for hcp forecasting",
                "Show me model drift for the ensemble models for hcp forecasting",
                "Compare ensemble vs base models for hcp forecasting",
                "Compare ensemble vs base models for hcp forecasting",
                "Compare ensemble vs base models for hcp forecasting",
                "Why is the ensemble performing worse than XGBoost?",
                "Which models have shown performance degradation over the last 6 months? Show me the trend in RMSE",
                "Which models have shown performance degradation over the last 6 months? Show me the trend in RMSE",
                "Which models have shown performance degradation over the last 6 months? Show me the trend in RMSE",
                "Compare the uplift predictions from messaging optimization models. Which HCP segments show the highest incremental lift from marketing campaigns? What features drive this?",
                "Compare the uplift predictions from messaging optimization models. Which HCP segments show the highest incremental lift from marketing campaigns? What features drive this?",
                "Compare the uplift predictions from messaging optimization models. Which HCP segments show the highest incremental lift from marketing campaigns? What features drive this?",
                "Compare drift thresholds used across different executions",
                "Compare drift thresholds used across different executions",
                "Compare drift thresholds used across different executions",
                "List all models with RMSE less than 50",
                "List all models with RMSE less than 50",
                "List all models with RMSE less than 50",
                "What's driving the difference between ensemble and base models?",
                "What's driving the difference between ensemble and base models?",
                "What's driving the difference between ensemble and base models?",
                "Compare Random Forest, XGBoost, and LightGBM",
                "Is there drift in the NRx forecasting models?",
                "Is there drift in the NRx forecasting models?",
                "Is there drift in the NRx forecasting models?",
            ]
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_queries, indent=2, fp=f)
        
        print(f"Created default config: {self.config_file}")
    
    def initialize_agent_state(self, query: str) -> Dict[str, Any]:
        """Create fresh agent state for query"""
        return {
            "messages": [],
            "user_query": query,
            "parsed_intent": None,
            "use_case": None,
            "models_requested": None,
            "comparison_type": None,
            "time_range": None,
            "metrics_requested": None,
            "entities_requested": None,
            "requires_visualization": False,
            "context_documents": None,
            "generated_sql": None,
            "sql_purpose": None,
            "expected_columns": None,
            "sql_retry_count": 0,
            "needs_sql_retry": False,
            "sql_error_feedback": None,
            "retrieved_data": None,
            "tool_calls": None,
            "analysis_results": None,
            "visualization_specs": None,
            "rendered_charts": None,
            "final_insights": None,
            "needs_clarification": False,
            "clarification_question": None,
            "loop_count": 0,
            "next_action": None,
            "execution_path": [],
            "conversation_context": {},
            "mentioned_models": [],
            "mentioned_model_ids": [],
            "last_query_summary": None,
            "current_topic": None,
            "viz_strategy": None,
            "viz_reasoning": None,
            "viz_warnings": None,
            "clarification_attempts": 0,
            "personalized_business_context": ""
        }
    
    def extract_all_sql_queries(self, all_states: List[Dict[str, Any]], test_id: str) -> List[Dict[str, Any]]:
        """Extract ALL SQL queries including retries from state history"""
        sql_attempts = []
        attempt_number = 0
        
        for state in all_states:
            generated_sql = state.get('generated_sql')
            
            # Only record if SQL actually changed (avoid duplicates)
            if generated_sql:
                # Check if this is a new SQL query
                is_new_sql = True
                if sql_attempts:
                    last_sql = sql_attempts[-1]['sql_query']
                    if last_sql == generated_sql:
                        is_new_sql = False
                
                if is_new_sql:
                    attempt_number += 1
                    sql_purpose = state.get('sql_purpose', '')
                    retry_count = state.get('sql_retry_count', 0)
                    needs_retry = state.get('needs_sql_retry', False)
                    error_feedback = state.get('sql_error_feedback', '')
                    
                    sql_attempts.append({
                        'test_id': test_id,
                        'attempt_number': attempt_number,
                        'retry_count': retry_count,
                        'sql_query': generated_sql,
                        'sql_purpose': sql_purpose,
                        'needs_retry': needs_retry,
                        'error_feedback': error_feedback if error_feedback else '',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        return sql_attempts
    
    def extract_sql_execution_results(self, messages: List[Any]) -> Tuple[bool, int, str]:
        """Extract SQL execution results from messages - get LAST execution result"""
        last_result = None
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    # Keep updating to get the LAST result (in case of retries)
                    last_result = result
                except Exception as e:
                    print(f"    Warning: Failed to parse ToolMessage: {e}")
                    continue
        
        if last_result:
            success = last_result.get('success', False)
            row_count = last_result.get('row_count', 0)
            error = last_result.get('error', '')
            return success, row_count, error
        
        return False, 0, 'No SQL execution found'
    
    def save_visualizations(self, rendered_charts: List[Dict], test_id: str) -> List[str]:
        """Save visualizations as HTML files and return paths"""
        saved_paths = []
        
        if not rendered_charts:
            return saved_paths
        
        for i, chart in enumerate(rendered_charts, 1):
            try:
                fig = chart.get('figure')
                if fig is None:
                    continue
                
                title = chart.get('title', f'Chart_{i}')
                safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
                safe_title = safe_title.replace(' ', '_')[:50]
                
                filename = f"{test_id}_chart{i}_{safe_title}.html"
                filepath = self.plots_dir / filename
                
                fig.write_html(str(filepath))
                
                # Store relative path from test_results directory
                relative_path = filepath.relative_to(self.results_dir)
                saved_paths.append(str(relative_path).replace('\\', '/'))
                
                print(f"    Saved plot {i}: {filename}")
                
            except Exception as e:
                print(f"    Error saving chart {i}: {e}")
                continue
        
        return saved_paths
    
    def extract_final_insights(self, final_state: Dict[str, Any], all_messages: List[Any]) -> str:
        """Extract final insights from state or messages"""
        if final_state.get('final_insights'):
            return final_state['final_insights']
        
        ai_messages = [msg for msg in all_messages if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_msg = ai_messages[-1]
            if hasattr(last_ai_msg, 'content') and last_ai_msg.content:
                if not ("Generated SQL query" in last_ai_msg.content or 
                        "execute_sql_query" in str(last_ai_msg.content)):
                    return last_ai_msg.content
        
        return "No insights generated"
    
    def run_single_query(self, query: str, test_id: str, allow_clarification: bool = True) -> Dict[str, Any]:
        """Run a single test query through the agent with full tracking and clarification handling"""
        print(f"\n{'='*80}")
        print(f"Testing: {test_id}")
        print(f"Query: {query}")
        print('='*80)
        
        start_time = time.time()
        
        # Initialize result structure
        result = {
            'Test_ID': test_id,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'User_Query': query,
            'Status': 'Unknown',
            'Processing_Time_Sec': 0,
            'Execution_Path': '',
            'Total_SQL_Attempts': 0,
            'SQL_Retry_Count': 0,
            'Final_SQL_Query': '',
            'SQL_Success': 'N',
            'Rows_Retrieved': 0,
            'SQL_Error': '',
            'Insights_Generated': '',
            'Num_Charts': 0,
            'Plot_File_Paths': '',
            'Needs_Clarification': 'N',
            'Clarification_Question': '',
            'Clarification_Provided': '',
            'Follow_Up_Queries': 0,
            'Error_Message': ''
        }
        
        try:
            # Initialize state
            current_query = query
            follow_up_count = 0
            max_follow_ups = 3
            
            # Track all states across all iterations
            all_states = []
            all_messages = []
            final_state = None
            
            # Loop to handle clarifications
            while follow_up_count <= max_follow_ups:
                print(f"\n[Iteration {follow_up_count + 1}] Processing query: {current_query[:80]}...")
                
                # Create state for current query
                iteration_state = self.initialize_agent_state(current_query)
                
                # If this is a follow-up, preserve conversation context
                if follow_up_count > 0 and final_state:
                    # Preserve context from previous iteration
                    iteration_state['messages'] = all_messages.copy()
                    iteration_state['conversation_context'] = final_state.get('conversation_context', {})
                    iteration_state['mentioned_models'] = final_state.get('mentioned_models', [])
                    iteration_state['current_topic'] = final_state.get('current_topic')
                    iteration_state['last_query_summary'] = final_state.get('last_query_summary')
                
                # Execute through graph and collect states
                iteration_states = []
                iteration_messages = []
                
                for event in graph.stream(iteration_state):
                    for node_name, value in event.items():
                        iteration_states.append(value)
                        if 'messages' in value:
                            iteration_messages.extend(value['messages'])
                        final_state = value
                
                # Accumulate all states and messages
                all_states.extend(iteration_states)
                all_messages.extend(iteration_messages)
                
                # Check if clarification is needed
                if final_state and final_state.get('needs_clarification', False):
                    clarification_question = final_state.get('clarification_question', '')
                    
                    result['Needs_Clarification'] = 'Y'
                    result['Clarification_Question'] = clarification_question
                    
                    print(f"\n{'='*80}")
                    print(f"🤔 CLARIFICATION NEEDED")
                    print(f"{'='*80}")
                    print(f"\n{clarification_question}")
                    print(f"\n{'='*80}")
                    
                    if allow_clarification and follow_up_count < max_follow_ups:
                        # Ask user for clarification
                        print(f"\nOptions:")
                        print(f"  1. Provide clarification")
                        print(f"  2. Skip this query")
                        print(f"  3. Auto-continue (use best guess)")
                        
                        choice = input(f"\nYour choice (1/2/3): ").strip()
                        
                        if choice == '1':
                            clarification_response = input(f"\nYour clarification: ").strip()
                            if clarification_response:
                                current_query = clarification_response
                                result['Clarification_Provided'] = clarification_response
                                follow_up_count += 1
                                print(f"\n✓ Continuing with clarification...")
                                continue
                            else:
                                print(f"\n⚠ Empty response, skipping query...")
                                break
                        elif choice == '2':
                            print(f"\n⏭ Skipping query...")
                            result['Status'] = 'Skipped'
                            break
                        elif choice == '3':
                            print(f"\n🤖 Auto-continuing without clarification...")
                            # Set clarification_attempts high to force continuation
                            iteration_state['clarification_attempts'] = 5
                            follow_up_count += 1
                            continue
                        else:
                            print(f"\n⚠ Invalid choice, skipping query...")
                            break
                    else:
                        print(f"\n⚠ Max clarifications reached or clarification disabled, stopping...")
                        break
                else:
                    # No clarification needed, exit loop
                    break
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result['Processing_Time_Sec'] = round(processing_time, 2)
            
            if final_state:
                # Extract execution path
                execution_path = final_state.get('execution_path', [])
                result['Execution_Path'] = ' → '.join(execution_path)
                
                # Extract ALL SQL queries (including retries)
                sql_attempts = self.extract_all_sql_queries(all_states, test_id)
                result['Total_SQL_Attempts'] = len(sql_attempts)
                
                # Store SQL queries in separate tracking
                self.sql_queries.extend(sql_attempts)
                
                # Get final SQL info
                if sql_attempts:
                    last_sql = sql_attempts[-1]
                    result['Final_SQL_Query'] = last_sql['sql_query']
                    result['SQL_Retry_Count'] = last_sql['retry_count']
                
                # Extract SQL execution results
                sql_success, rows_retrieved, sql_error = self.extract_sql_execution_results(all_messages)
                result['SQL_Success'] = 'Y' if sql_success else 'N'
                result['Rows_Retrieved'] = rows_retrieved
                result['SQL_Error'] = sql_error if not sql_success else ''
                
                # Extract insights
                result['Insights_Generated'] = self.extract_final_insights(final_state, all_messages)
                
                # Extract and save visualizations
                rendered_charts = final_state.get('rendered_charts', [])
                if rendered_charts:
                    plot_paths = self.save_visualizations(rendered_charts, test_id)
                    result['Num_Charts'] = len(rendered_charts)
                    result['Plot_File_Paths'] = '\n'.join(plot_paths)
                
                # Check for clarification
                needs_clarification = final_state.get('needs_clarification', False)
                result['Needs_Clarification'] = 'Y' if needs_clarification else 'N'
                clarification_q = final_state.get('clarification_question', '')
                result['Clarification_Question'] = clarification_q if clarification_q else ''
                
                # Determine status
                if result['Insights_Generated'] and result['Insights_Generated'] != "No insights generated":
                    if sql_success or needs_clarification:
                        result['Status'] = 'Success'
                    else:
                        result['Status'] = 'Partial'
                else:
                    result['Status'] = 'Failed'
                
                # Store detailed execution info
                self.execution_details.append({
                    'test_id': test_id,
                    'execution_path': execution_path,
                    'num_states': len(all_states),
                    'num_messages': len(all_messages),
                    'final_state_keys': list(final_state.keys()),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                print(f"✓ Status: {result['Status']}")
                print(f"✓ Processing Time: {result['Processing_Time_Sec']}s")
                print(f"✓ SQL Attempts: {result['Total_SQL_Attempts']} (Retries: {result['SQL_Retry_Count']})")
                print(f"✓ SQL Success: {result['SQL_Success']}")
                print(f"✓ Rows Retrieved: {result['Rows_Retrieved']}")
                print(f"✓ Charts Generated: {result['Num_Charts']}")
            else:
                result['Status'] = 'Failed'
                result['Error_Message'] = 'No final state returned from graph'
                print(f"✗ Failed: No final state")
        
        except Exception as e:
            processing_time = time.time() - start_time
            result['Processing_Time_Sec'] = round(processing_time, 2)
            result['Status'] = 'Failed'
            result['Error_Message'] = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"✗ Failed with error: {str(e)}")
        
        return result
    
    def run_all_tests(self):
        """Run all test queries and collect comprehensive results"""
        print("\n" + "="*80)
        print("PHARMA ANALYTICS AGENT - ENHANCED AUTOMATED TESTING")
        print("="*80)
        
        try:
            initialize_connection_pool()
            print("✓ Database connection initialized")
        except Exception as e:
            print(f"✗ Database initialization failed: {e}")
            return
        
        # Load queries
        queries = self.load_test_queries()
        
        print(f"\nTest Run ID: {self.test_run_id}")
        print(f"Total Queries: {len(queries)}")
        print(f"Results Directory: {self.run_dir}")
        print("\nStarting tests...\n")
        
        # Run each query
        for i, query in enumerate(queries, 1):
            test_id = f"TEST_{i:03d}"
            result = self.run_single_query(query, test_id)
            self.test_results.append(result)
            
            # Save progress after each query
            self.save_all_results(interim=True)
            
            print(f"\nProgress: {i}/{len(queries)} queries completed")
        
        # Final save and summary
        self.save_all_results(interim=False)
        self.print_summary()
        
        # Cleanup
        try:
            close_connection_pool()
            print("\n✓ Database connection closed")
        except:
            pass
    
    def save_all_results(self, interim: bool = False):
        """Save all results to multiple Excel sheets"""
        if not self.test_results:
            return
        
        excel_filename = self.run_dir / f"test_results_{self.test_run_id}.xlsx"
        
        try:
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # Sheet 1: Main Test Results
                df_main = pd.DataFrame(self.test_results)
                
                # Add summary row if final save
                if not interim and len(self.test_results) > 0:
                    summary = {
                        'Test_ID': 'SUMMARY',
                        'Timestamp': '',
                        'User_Query': f"Total: {len(self.test_results)} queries",
                        'Status': f"Success: {sum(1 for r in self.test_results if r['Status'] == 'Success')} | "
                                  f"Partial: {sum(1 for r in self.test_results if r['Status'] == 'Partial')} | "
                                  f"Failed: {sum(1 for r in self.test_results if r['Status'] == 'Failed')}",
                        'Processing_Time_Sec': round(sum(r['Processing_Time_Sec'] for r in self.test_results) / len(self.test_results), 2),
                        'Execution_Path': '',
                        'Total_SQL_Attempts': sum(r['Total_SQL_Attempts'] for r in self.test_results),
                        'SQL_Retry_Count': sum(r['SQL_Retry_Count'] for r in self.test_results),
                        'Final_SQL_Query': f"Total attempts: {sum(r['Total_SQL_Attempts'] for r in self.test_results)}",
                        'SQL_Success': f"{sum(1 for r in self.test_results if r['SQL_Success'] == 'Y')}/{len(self.test_results)}",
                        'Rows_Retrieved': sum(r['Rows_Retrieved'] for r in self.test_results),
                        'SQL_Error': '',
                        'Insights_Generated': '',
                        'Num_Charts': sum(r['Num_Charts'] for r in self.test_results),
                        'Plot_File_Paths': '',
                        'Needs_Clarification': f"{sum(1 for r in self.test_results if r['Needs_Clarification'] == 'Y')} queries",
                        'Clarification_Question': '',
                        'Error_Message': ''
                    }
                    df_main = pd.concat([df_main, pd.DataFrame([summary])], ignore_index=True)
                
                df_main.to_excel(writer, sheet_name='Test Results', index=False)
                self._format_worksheet(writer.sheets['Test Results'], df_main)
                
                # Sheet 2: SQL Query History (ALL queries including retries)
                if self.sql_queries:
                    df_sql = pd.DataFrame(self.sql_queries)
                    # Sort by test_id and attempt_number for clarity
                    df_sql = df_sql.sort_values(['test_id', 'attempt_number'])
                    df_sql.to_excel(writer, sheet_name='SQL Query History', index=False)
                    self._format_worksheet(writer.sheets['SQL Query History'], df_sql)
                
                # Sheet 3: Execution Details
                if self.execution_details:
                    df_exec = pd.DataFrame(self.execution_details)
                    # Convert list columns to strings for Excel
                    if 'execution_path' in df_exec.columns:
                        df_exec['execution_path'] = df_exec['execution_path'].apply(
                            lambda x: ' → '.join(x) if isinstance(x, list) else str(x)
                        )
                    if 'final_state_keys' in df_exec.columns:
                        df_exec['final_state_keys'] = df_exec['final_state_keys'].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                        )
                    df_exec.to_excel(writer, sheet_name='Execution Details', index=False)
                    self._format_worksheet(writer.sheets['Execution Details'], df_exec)
                
                # Sheet 4: SQL Retry Analysis
                if self.sql_queries:
                    retry_analysis = []
                    for test in self.test_results:
                        test_id = test['Test_ID']
                        test_sqls = [s for s in self.sql_queries if s['test_id'] == test_id]
                        if test_sqls:
                            retry_analysis.append({
                                'Test_ID': test_id,
                                'Query': test['User_Query'][:100] + '...' if len(test['User_Query']) > 100 else test['User_Query'],
                                'Total_Attempts': len(test_sqls),
                                'Max_Retry_Count': max(s['retry_count'] for s in test_sqls),
                                'Final_Success': test['SQL_Success'],
                                'Final_Rows': test['Rows_Retrieved'],
                                'Had_Errors': 'Y' if any(s['error_feedback'] for s in test_sqls) else 'N'
                            })
                    
                    if retry_analysis:
                        df_retry = pd.DataFrame(retry_analysis)
                        df_retry.to_excel(writer, sheet_name='SQL Retry Analysis', index=False)
                        self._format_worksheet(writer.sheets['SQL Retry Analysis'], df_retry)
            
            status = "interim" if interim else "final"
            print(f"\n✓ Results saved ({status}): {excel_filename.name}")
        
        except Exception as e:
            print(f"\n✗ Error saving results: {e}")
            traceback.print_exc()
    
    def _format_worksheet(self, worksheet, df):
        """Auto-adjust column widths in worksheet"""
        try:
            from openpyxl.utils import get_column_letter
            
            for i, col in enumerate(df.columns, 1):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                max_length = min(max_length, 100)
                
                col_letter = get_column_letter(i)
                worksheet.column_dimensions[col_letter].width = max_length + 2
        except:
            pass
    
    def print_summary(self):
        """Print comprehensive test summary"""
        if not self.test_results:
            return
        
        total = len(self.test_results)
        success = sum(1 for r in self.test_results if r['Status'] == 'Success')
        partial = sum(1 for r in self.test_results if r['Status'] == 'Partial')
        failed = sum(1 for r in self.test_results if r['Status'] == 'Failed')
        
        avg_time = sum(r['Processing_Time_Sec'] for r in self.test_results) / total
        total_sql_attempts = sum(r['Total_SQL_Attempts'] for r in self.test_results)
        total_retries = sum(r['SQL_Retry_Count'] for r in self.test_results)
        sql_success = sum(1 for r in self.test_results if r['SQL_Success'] == 'Y')
        total_charts = sum(r['Num_Charts'] for r in self.test_results)
        
        # Count queries that needed retries
        queries_with_retries = sum(1 for r in self.test_results if r['SQL_Retry_Count'] > 0)
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"\nTotal Queries Tested: {total}")
        print(f"  ✓ Success: {success} ({success/total*100:.1f}%)")
        print(f"  ⚠ Partial: {partial} ({partial/total*100:.1f}%)")
        print(f"  ✗ Failed: {failed} ({failed/total*100:.1f}%)")
        
        print(f"\nSQL Query Statistics:")
        print(f"  Total SQL Attempts: {total_sql_attempts}")
        print(f"  Total Retries: {total_retries}")
        print(f"  Queries Requiring Retries: {queries_with_retries} ({queries_with_retries/total*100:.1f}%)")
        print(f"  SQL Execution Success: {sql_success}/{total} ({sql_success/total*100:.1f}%)")
        print(f"  Average Attempts per Query: {total_sql_attempts/total:.1f}")
        
        print(f"\nPerformance:")
        print(f"  Average Processing Time: {avg_time:.2f}s")
        print(f"  Total Visualizations: {total_charts} charts")
        
        if failed > 0:
            print(f"\nFailed Queries:")
            for r in self.test_results:
                if r['Status'] == 'Failed':
                    print(f"  - {r['Test_ID']}: {r['User_Query'][:60]}...")
                    if r['SQL_Error']:
                        print(f"    SQL Error: {r['SQL_Error'][:100]}")
        
        if queries_with_retries > 0:
            print(f"\nQueries with SQL Retries:")
            for r in self.test_results:
                if r['SQL_Retry_Count'] > 0:
                    print(f"  - {r['Test_ID']}: {r['SQL_Retry_Count']} retries - {r['User_Query'][:50]}...")
        
        print("\n" + "="*80)
        print(f"Results saved to: {self.run_dir}")
        print(f"Excel file: test_results_{self.test_run_id}.xlsx")
        print(f"Plots directory: {self.plots_dir}")
        print("="*80 + "\n")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("ENHANCED AUTOMATED TESTING - PHARMA ANALYTICS AGENT")
    print("="*80)
    print("\nThis script will:")
    print("1. Load test queries from test_queries.json")
    print("2. Run each query through the agent")
    print("3. Track ALL SQL queries (including retries)")
    print("4. Capture comprehensive execution state")
    print("5. Save visualizations as HTML files")
    print("6. Generate multi-sheet Excel report with:")
    print("   - Main test results")
    print("   - SQL query history (all attempts)")
    print("   - Execution details")
    print("   - SQL retry analysis")
    print("\nResults saved after each query to prevent data loss.")
    
    response = input("\nProceed with testing? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nTesting cancelled.")
        return
    
    try:
        tester = EnhancedAgentTester(config_file="test_queries.json")
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        print("Partial results have been saved.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()