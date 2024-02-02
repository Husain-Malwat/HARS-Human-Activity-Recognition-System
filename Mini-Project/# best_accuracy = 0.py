# best_accuracy = 0
# best_hyperparameters = {}

# out = {}
# count = 0
# for max_depth in tqdm(hyperparameters['max_depth']):
#     for min_samples_split in hyperparameters['min_samples_split']:
#         for criterion in hyperparameters['criteria_values']:
#             # Create and fit the decision tree classifier with the current hyperparameters
#             Recognizer = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=42)
#             Recognizer.fit(X_train, y_train)
            
#             # Evaluate the performance on the validation set
#             val_accuracy = Recognizer.score(X_val, y_val)
#             out[count] = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'criterion': criterion, 'val_accuracy': val_accuracy}
#             count += 1

# hparam_df = pd.DataFrame(out).T
# best_hyperparameters_row_id = 0
# for i in tqdm(range(0,len(hyperparameters['max_depth'])*len(hyperparameters['min_samples_split'])*len(hyperparameters['criteria_values']))):
#     if (hparam_df['val_accuracy'][i] > hparam_df['val_accuracy'][best_hyperparameters_row_id]):
#         best_hyperparameters_row_id = i

# best_accuracy = hparam_df['val_accuracy'][best_hyperparameters_row_id]
# best_hyperparameters = {'max_depth':hparam_df['max_depth'][best_hyperparameters_row_id], 'min_samples_split':hparam_df['min_samples_split'][best_hyperparameters_row_id], 'criterion':hparam_df['criterion'][best_hyperparameters_row_id] }
# print("Best Hyperparameters:", best_hyperparameters)
# print("Validation Set accuracy: {:.4f}".format(best_accuracy))


