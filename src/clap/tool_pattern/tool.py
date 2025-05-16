import json
import inspect
import functools
from typing import Callable, Any, Dict, List 
import anyio
import jsonschema 


def get_fn_signature(fn: Callable) -> dict:
   
    type_mapping = {
        "int": "integer", "str": "string", "bool": "boolean",
        "float": "number", "list": "array", "dict": "object",
    }
    parameters = {"type": "object", "properties": {}, "required": []}
    sig = inspect.signature(fn)
    for name, type_hint in fn.__annotations__.items():
        if name == "return": continue
        param_type_name = getattr(type_hint, "__name__", str(type_hint))
        schema_type = type_mapping.get(param_type_name.lower(), "string") 
        parameters["properties"][name] = {"type": schema_type}
        if sig.parameters[name].default is inspect.Parameter.empty:
            parameters["required"].append(name)
    if not parameters.get("required"): 
        if "required" in parameters: del parameters["required"]
    return {"type": "function", "function": {"name": fn.__name__, "description": fn.__doc__, "parameters": parameters}}


def validate_and_coerce_arguments(tool_call_args: Dict[str, Any], tool_schema: Dict[str, Any]) -> Dict[str, Any]: # Renamed for clarity
    """
    Validates and coerces arguments in the input dictionary based on the tool's JSON schema.
    Then, performs final validation using jsonschema.

    Args:
        tool_call_args (Dict[str, Any]): Arguments from LLM (often strings).
        tool_schema (Dict[str, Any]): Tool's JSON schema.

    Returns:
        Dict[str, Any]: Arguments with values coerced to correct types.

    Raises:
        jsonschema.ValidationError: If final validation fails after coercion.
        ValueError: If coercion fails for a required argument or types are incompatible.
    """
    parameter_schema = tool_schema.get("function", {}).get("parameters", {})
    properties = parameter_schema.get("properties", {})
    required_args = parameter_schema.get("required", [])
    
    coerced_args: Dict[str, Any] = {}

    
    string_coercion_map = {
        "integer": int,
        "number": float, 
        "boolean": lambda x: x.lower() in ['true', '1', 'yes'] if isinstance(x, str) else bool(x),
        "string": str,
        
        "array": lambda x: json.loads(x) if isinstance(x, str) else x,
        "object": lambda x: json.loads(x) if isinstance(x, str) else x,
    }
    
    expected_python_type_map = {
        "integer": int, "number": (int, float), "boolean": bool,
        "string": str, "array": list, "object": dict,
    }

    for arg_name, arg_value in tool_call_args.items():
        prop_schema = properties.get(arg_name)
        if not prop_schema:
           
            coerced_args[arg_name] = arg_value 
            continue

        expected_schema_type = prop_schema.get("type")
        python_type_tuple = expected_python_type_map.get(expected_schema_type) 

        
        if python_type_tuple and isinstance(arg_value, python_type_tuple):
            coerced_args[arg_name] = arg_value
            continue

        
        coercer = string_coercion_map.get(expected_schema_type) 
        if coercer:
            try:
                coerced_value = coercer(arg_value)
                
                if expected_schema_type in ["array", "object"] and python_type_tuple:
                    if not isinstance(coerced_value, python_type_tuple): 
                         raise ValueError(f"Decoded JSON for '{arg_name}' is not expected type '{expected_schema_type}'.")
                coerced_args[arg_name] = coerced_value
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                
                
                if arg_name in required_args:
                    raise ValueError(
                        f"Error coercing required argument '{arg_name}' with value '{arg_value}' "
                        f"to type '{expected_schema_type}': {e}"
                    )
                else:
                    coerced_args[arg_name] = arg_value 
        else:
            
            coerced_args[arg_name] = arg_value

   
    try:
        jsonschema.validate(instance=coerced_args, schema=parameter_schema)
    except jsonschema.ValidationError as e:
       
        raise

    return coerced_args


class Tool:
    def __init__(self, name: str, fn: Callable, fn_schema: dict):
        self.name = name
        self.fn = fn
        self.fn_schema = fn_schema
        # self.fn_signature = json.dumps(fn_schema) 

    def __str__(self):
        return json.dumps(self.fn_schema, indent=2)

    async def run(self, **kwargs: Any) -> Any: 
        """
        Executes the tool, validating and coercing arguments first.
        """
        function_name = self.fn_schema.get("function", {}).get("name", "unknown_tool")
        try:
           
            validated_and_coerced_kwargs = validate_and_coerce_arguments(kwargs, self.fn_schema)
        except (jsonschema.ValidationError, ValueError) as e:
           
            return f"Error: Invalid arguments for tool {function_name} - {str(e)}"
        except Exception as e:
             return f"Error: Unexpected issue with arguments for tool {function_name}."

        
        try:
            if inspect.iscoroutinefunction(self.fn):
                return await self.fn(**validated_and_coerced_kwargs)
            else:
                
                func_with_args = functools.partial(self.fn, **validated_and_coerced_kwargs)
                return await anyio.to_thread.run_sync(func_with_args)
        except Exception as e:
            
             return f"Error executing tool {function_name}: {e}"


def tool(fn: Callable):
    fn_schema = get_fn_signature(fn)
    if not fn_schema or 'function' not in fn_schema or 'name' not in fn_schema['function']:
            raise ValueError(f"Could not generate valid schema for function {fn.__name__}")
    return Tool(name=fn_schema["function"]["name"], fn=fn, fn_schema=fn_schema)

