from torch_geometric.data import HeteroData
import pandas as pd
import torch

user_train_columns = ['current_age','retirement_age','birth_year','birth_month','is_female','zipcode','latitude','longitude','per_capita_income_zipcode','yearly_income_person',
                      'total_debt','fico_score','num_credit_cards']

user_transaction_history_columns = ['year','month','day','unix_time','amount','card_reader','city_code','state_code',
                                    'err_tech_glitch', 'err_insufficient_balance', 'err_bad_pin', 'err_bad_expiration', 'err_bad_card_number', 
                                    'err_bad_cvv', 'err_bad_zip_code']
user_cards_column = ['card_number_group','expires_enc','has_chip','cards_issued','credit_limit','acct_open_date_enc','year_pin_changed','card_brand_id','card_type_id']

class GraphBuilder:
    @staticmethod
    def _get_user_cards(user_id, cards):
        user_cards = cards[cards['user']==user_id]
        user_cards = user_cards.reset_index(drop=True)
        #set_trace()
        return user_cards

    @staticmethod
    def _get_user_history_transactions(user_id, current_transaction, max_transactions=10):
        unix_time = current_transaction['unix_time']
        user_transactions = transactions[transactions['user_id']==user_id & transactions['unix_time'] < unix_time]
        user_transactions = user_transactions.reset_index(drop=True)
        #set_trace()
        return user_transactions[:max_transactions]

    @staticmethod
    def _get_user_history(user_id, user_transaction_history):
        if len(user_transaction_history) == 0:
            return [0.0] * 8
        
        amount_sum = user_transaction_history['amount'].sum()
        is_fraud_sum = user_transaction_history['is_fraud'].sum()
    
        error_columns = [col for col in user_transaction_history.columns if col.startswith('err_')]
        error_rows_count = user_transaction_history[error_columns].any(axis=1).sum()
    
        unix_time_diffs = user_transaction_history['unix_time'].diff(-1).fillna(-1).iloc[:5].tolist()
        unix_time_diffs += [-1] * (5 - len(unix_time_diffs))
    
        #set_trace()
        return [[amount_sum, is_fraud_sum, error_rows_count] + unix_time_diffs]

    @staticmethod
    def _get_user_owns_card_edges(user,user_cards):
        zeros_tensor = torch.zeros(len(user_cards), dtype=torch.long)
        range_tensor = torch.arange(0, len(user_cards), dtype=torch.long)
    
        #set_trace()
        return torch.stack([zeros_tensor, range_tensor])

    @staticmethod
    def _get_transactions_in_user_history_edges(user_history_transactions):
        zeros_tensor = torch.zeros(len(user_history_transactions),dtype=torch.long)
        range_tensor = torch.arange(0, len(user_history_transactions),dtype=torch.long)
    
        #set_trace()
        return torch.stack([range_tensor, zeros_tensor])[:user_history_model_max_size]

    @staticmethod
    def _get_transactions_paid_with_card_edges(user_history_transactions, user_cards):
        history = user_history_transactions.reset_index(drop=True).reset_index().rename(columns={'index':'transaction_index'})
        cards = user_cards.reset_index(drop=True).reset_index().rename(columns={'index':'card_set_index'})
        
        merged_set = history.merge(cards, left_on='card_id', right_on='card_set_index', how='inner')
        #set_trace()
        return torch.stack([
            torch.tensor(merged_set['transaction_index'].to_numpy(), dtype=torch.long),
            torch.tensor(merged_set['card_set_index'].to_numpy(), dtype=torch.long)
        ])

    @staticmethod
    def _get_merchant_features(merchants, transaction_history):
        amounts = transaction_history.groupby('merchant_name')['amount'].sum().reset_index()
        result = pd.merge(merchants, amounts, on='merchant_name', how='left')
        result.fillna(0, inplace=True)
        #set_trace()
        return result

    @staticmethod
    def _get_transaction_made_at_merchant_edges(user_history_transactions, merchants):
        indexed_merchants = merchants.reset_index(drop=True).reset_index().rename(columns={'index':'merchant_index'})
        indexed_user_history_transactions = user_history_transactions.reset_index(drop=True).reset_index().rename(columns={'index':'transaction_index'})
        
        merged = indexed_user_history_transactions.merge(indexed_merchants, on='merchant_name', how='inner')
        
        history_indexes = merged['transaction_index']
        merchants_indexes = merged['merchant_index']
        
        edges = torch.stack([
            torch.tensor(history_indexes.to_numpy(), dtype=torch.long),
            torch.tensor(merchants_indexes.to_numpy(), dtype=torch.long)])
        #print(f'edge_count={len(edges)}, history_indexes = {history_indexes.to_numpy()}')
        #set_trace()
        return edges

    @staticmethod
    def _get_merchant_selling_pending_transaction(merchants, current_transaction):
        current_merchant_name = current_transaction['merchant_name']
        
        index = merchants[merchants['merchant_name']==current_merchant_name].index
        #set_trace()
        return torch.stack([
            torch.tensor(index.to_numpy(), dtype=torch.long),
            torch.zeros(len(index), dtype=torch.long)
        ])

    @staticmethod
    def generate_transaction_graph(user_id, current_transaction, user_history, cards, users):
        graph = HeteroData()
        
        user_history_transactions = user_history[-user_history_model_max_size:]
        user_cards = get_user_cards(user_id, cards)
        user = users.loc[users['user_id']==user_id, user_train_columns]
        #total_transactions = user_history_transactions
        merchants = pd.concat([user_history_transactions, pd.DataFrame([current_transaction])])['merchant_name'].unique()
        merchants = pd.DataFrame(merchants, columns=['merchant_name'])
        
        graph['user'].x = torch.tensor(user.values, dtype=torch.float)
        graph['card'].x = torch.tensor(user_cards.values, dtype=torch.float)    
        graph['user_history'].x = torch.tensor(get_user_history(user_id, user_history), dtype=torch.float)    
        graph['user_history_transaction'].x = torch.tensor(user_history_transactions[(user_transaction_history_columns + ['is_fraud'])].values, dtype=torch.float)
        graph['pending_transaction'].x = torch.tensor(current_transaction[user_transaction_history_columns].values, dtype=torch.float)
        graph['merchant'].x = torch.tensor(get_merchant_features(merchants, user_history_transactions).values, dtype=torch.float)
    
        edges = get_user_owns_card_edges(user,user_cards)
        opposite_edges = torch.stack([edges[1], edges[0]], dim=0)
        graph['user','owns','card'].edge_index = edges
        graph['card','belongs_to','user'].edge_index = opposite_edges
        
        graph['user','has','user_history'].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        graph['user_history','belongs_to','user'].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        graph['user_history_transaction','part_of','user_history'].edge_index = get_transactions_in_user_history_edges(user_history_transactions)
    
        edges = get_transactions_paid_with_card_edges(user_history_transactions, user_cards)
        opposite_edges = torch.stack([edges[1], edges[0]], dim=0)
        graph['user_history_transaction','paid_with','card'].edge_index = edges
        graph['card','paid_for','user_history_transaction'].edge_index = opposite_edges
        
        edges = get_transaction_made_at_merchant_edges(user_history_transactions,merchants)
        opposite_edges = torch.stack([edges[1], edges[0]], dim=0)
        graph['user_history_transaction','made_at','merchant'].edge_index = edges
        graph['merchant', 'made', 'user_history_transaction'].edge_index = opposite_edges
    
        graph['user_history','reflects_on','pending_transaction'].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        graph['merchant','selling','pending_transaction'].edge_index = get_merchant_selling_pending_transaction(merchants, current_transaction)
        graph['user','purchasing','pending_transaction'].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        graph.y = current_transaction['is_fraud']
    
        #set_trace()
        return graph

    




    