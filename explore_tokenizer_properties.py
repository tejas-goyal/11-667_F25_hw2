#!/usr/bin/env python3
"""
Temporary exploration file for understanding tokenizer properties.
Test all three properties: injective, invertible, and concatenation preservation.
"""

from transformers import AutoTokenizer

def test_injective_property():
    """
    Injective: for all s != s', tokenize(s) != tokenize(s')
    Testing if different strings produce the same token sequence.
    """
    print("=" * 60)
    print("TESTING INJECTIVE PROPERTY")
    print("=" * 60)
    
    tokenizers = ['google-bert/bert-base-cased', 'gpt2', 't5-small']
    test_cases = [
        ('hello', 'hello '),           # trailing space
        ('', ' '),                     # empty vs space
        ('test\t', 'test '),          # tab vs space
        ('hi\n', 'hi'),               # newline vs none
        ('café', 'cafe'),             # accents
        ('don\'t', 'don \'t'),        # apostrophe handling
    ]
    
    for i, (s1, s2) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {repr(s1)} vs {repr(s2)}")
        print("-" * 50)
        
        for tokenizer_name in tokenizers:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, verbose=False)
                tokens1 = tokenizer.encode(s1, add_special_tokens=False)
                tokens2 = tokenizer.encode(s2, add_special_tokens=False)
                
                if s1 != s2 and tokens1 == tokens2:
                    print(f"  ✅ VIOLATION in {tokenizer_name}:")
                    print(f"     '{s1}' -> {tokens1}")
                    print(f"     '{s2}' -> {tokens2}")
                else:
                    print(f"  ✓ OK in {tokenizer_name}:")
                    print(f"     '{s1}' -> {tokens1}")
                    print(f"     '{s2}' -> {tokens2}")
            except Exception as e:
                print(f"  ❌ Error with {tokenizer_name}: {e}")

def test_invertible_property():
    """
    Invertible: for any string s, s = detokenize(tokenize(s))
    Testing if tokenize -> detokenize recovers original string.
    """
    print("\n" + "=" * 60)
    print("TESTING INVERTIBLE PROPERTY")
    print("=" * 60)
    
    tokenizers = ['google-bert/bert-base-cased', 'gpt2', 't5-small']
    test_cases = [
        "hello ",                      # trailing space
        "  multiple  spaces  ",        # multiple spaces
        "test\t",                     # tab character
        "hi\n",                       # newline
        "\u200b",                     # zero-width space
        "café",                       # unicode
        "",                           # empty string
    ]
    
    for i, s in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {repr(s)}")
        print("-" * 50)
        
        for tokenizer_name in tokenizers:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, verbose=False)
                tokens = tokenizer.encode(s, add_special_tokens=False)
                s_recovered = tokenizer.decode(tokens)
                
                if s != s_recovered:
                    print(f"  ✅ VIOLATION in {tokenizer_name}:")
                    print(f"     Original: {repr(s)}")
                    print(f"     Tokens:   {tokens}")
                    print(f"     Recovered: {repr(s_recovered)}")
                else:
                    print(f"  ✓ OK in {tokenizer_name}:")
                    print(f"     {repr(s)} -> {tokens} -> {repr(s_recovered)}")
            except Exception as e:
                print(f"  ❌ Error with {tokenizer_name}: {e}")

def test_concatenation_property():
    """
    Preserves concatenation: tokenize(s + s') = tokenize(s) + tokenize(s')
    Testing if tokenizing concatenated strings equals concatenated tokens.
    """
    print("\n" + "=" * 60)
    print("TESTING CONCATENATION PROPERTY")
    print("=" * 60)
    
    tokenizers = ['google-bert/bert-base-cased', 'gpt2', 't5-small']
    test_cases = [
        ('hello', 'world'),           # simple words
        ('un', 'happy'),              # subword boundary
        ('', 'test'),                 # empty + word
        ('test', ''),                 # word + empty
        ('pre', 'fix'),               # prefix-like
        ('new', 'york'),              # compound
        ('don', '\'t'),               # apostrophe split
    ]
    
    for i, (a, b) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {repr(a)} + {repr(b)}")
        print("-" * 50)
        
        for tokenizer_name in tokenizers:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, verbose=False)
                
                # Tokenize concatenated string
                tokens_concat = tokenizer.encode(a + b, add_special_tokens=False)
                
                # Tokenize separately and concatenate
                tokens_a = tokenizer.encode(a, add_special_tokens=False)
                tokens_b = tokenizer.encode(b, add_special_tokens=False)
                tokens_separate = tokens_a + tokens_b
                
                if tokens_concat != tokens_separate:
                    print(f"  ✅ VIOLATION in {tokenizer_name}:")
                    print(f"     tokenize('{a}' + '{b}') = {tokens_concat}")
                    print(f"     tokenize('{a}') + tokenize('{b}') = {tokens_a} + {tokens_b} = {tokens_separate}")
                else:
                    print(f"  ✓ OK in {tokenizer_name}:")
                    print(f"     Both ways give: {tokens_concat}")
            except Exception as e:
                print(f"  ❌ Error with {tokenizer_name}: {e}")

def interactive_test():
    """
    Interactive testing function - test your own strings!
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)
    print("Test your own strings! (type 'quit' to exit)")
    
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased', verbose=False)
    
    while True:
        try:
            s = input("\nEnter a string to test: ").strip()
            if s.lower() == 'quit':
                break
                
            tokens = tokenizer.encode(s, add_special_tokens=False)
            recovered = tokenizer.decode(tokens)
            
            print(f"Original:  {repr(s)}")
            print(f"Tokens:    {tokens}")
            print(f"Recovered: {repr(recovered)}")
            print(f"Invertible: {'✓ YES' if s == recovered else '✗ NO'}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("TOKENIZER PROPERTY EXPLORER")
    print("Understanding violations in popular tokenizers")
    
    # Run all tests
    test_injective_property()
    test_invertible_property()
    test_concatenation_property()
    
    # Optional interactive testing
    response = input("\nWould you like to test your own strings interactively? (y/n): ")
    if response.lower().startswith('y'):
        interactive_test()
    
    print("\nDone! You can run this script anytime: python explore_tokenizer_properties.py")