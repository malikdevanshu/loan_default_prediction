from loan_check.utils.utils import (
    build_feature_stages,
    get_config_values,
    get_model_paths,
    get_models,
    load_and_prepare_data,
)

def train_models():
    config_values = get_config_values()
    model_dir = config_values["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
 
    train, test = load_and_prepare_data()  # noqa: RUF059
    train = train.cache()
 
    feature_stages = build_feature_stages(train)
    models = get_models()
 
    for model_name, classifier_class in models.items():
        print(f"\nTraining model: {model_name}")
 
        model = classifier_class()
        model.train(train, feature_stages)
 
        model_path = get_model_paths(
            model_name=model_name,
            model_type="baseline",
        )
        model.save_model(model_path)
 
        print(f"Saved model to: {model_path}")
 
    print("\nTraining complete.")
 
 
if __name__ == "__main__":
    train_models()