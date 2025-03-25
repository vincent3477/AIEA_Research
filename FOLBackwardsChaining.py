

class KB:
    def __init__(self):
        self.facts = []
        self.rules = []
        self.verbose = False
    

    #make a new fact.
    #def make_new_fact(self, statement, target):
    #    self.facts.append((None, statement, target, None))
    
    def make_verbose(self):
        self.verbose = True
    #make a new fact (2-way relation). 
    def make_new_two_way_relation(self, name, x, y):
        return self.facts.append(("TWO_WAY", name, x, y))
    
    #make a new fact (1-way relation). x will have relationship r with y ONLY.
    def make_new_one_way_relation(self, name, x, y):
        return self.facts.append(("ONE_WAY", name, x, y))


    def make_new_rule(self, relation_type, add_fact, then_path):
        #rule includes AND and OR.
        if type(add_fact) != list:
            raise Exception("You need to pass in a list of facts.") 
        if relation_type.lower() == "and":
            #if all of the facts are true then the then_path is also true
            self.rules.append(("AND",add_fact,then_path))
        elif relation_type.lower() == "or":
             #if any of the facts are true then the then_path is also true
             self.rules.append(("OR",add_fact,then_path))


    def get_mykb(self):
        return self.facts, self.rules

    def _is_var(self, x):
        if isinstance(x, str)  and x.islower():
            return True
        return False

    def unify(self, t1, t2, substitution):
        if self.verbose:
            print("call to unify with the following terms: ")
            print(t1)
            print(t2)
        if t1 == t2:
            return
        if self._is_var(t1):
            substitution[t1] = t2
            return
        if self._is_var(t2):
            substitution[t2] = t1
            return

    def derive_conclusion(self, hypothesis, facts, rules, substitution):
        #hypothesis has to be of the form (two_way, name, x, y), (one_way, name, x y), or (x, y)
        n_way, name, x, y = hypothesis

        x = substitution.get(x,x)
        y = substitution.get(y,y)

        #print(x)
        #print(y)

        #check for a direct match first.
        for i in facts:
            if i == hypothesis:
                return True
        
        for j in facts:
            relation_type, relation_name, x1, y1 = j
            
            if relation_name != name or n_way != relation_type:
                continue
            if x == x1 and self._is_var(y):
                if self.verbose: print("unify y")
                self.unify(y, y1, substitution)
                return True
            if y == y1 and self._is_var(x):
                if self.verbose: print("unify x")
                self.unify(x, x1, substitution)
                return True

        for (rule_type, sub_facts, then_path) in rules:
            then_n_way, then_name, then_x, then_y = then_path
            self.unify(then_x, x, substitution)
            self.unify(then_y, y, substitution)
            if then_n_way == n_way and name == then_name:
                if rule_type == 'AND':
                    for sub_n_way, sub_relation, sub_x, sub_y in sub_facts:
                        subgoal = (sub_n_way, sub_relation, sub_x, sub_y)
                        if self.verbose:
                            print("\n\nstuff plugged into derive_conlusion")
                            print(f"we want to prove that: {subgoal}")
                            print(f"facts: {facts}")
                            print(f"rules: {rules}")
                            print(f"substitutions: {substitution}")
                        new_sub = substitution.copy()
                        if not self.derive_conclusion( subgoal, facts, rules, new_sub):
                            return False
                    return True
                if rule_type == 'OR':
                    for sub_n_way, sub_relation, sub_x, sub_y in sub_facts:
                        subgoal = (sub_n_way, sub_relation, sub_x, sub_y)
                        new_sub = substitution.copy()
                        if self.derive_conclusion( subgoal, facts, rules, new_sub):
                            return True
                    return False
                
            return False
        return False
                

if __name__ == "__main__":
    mykb = KB()
    #mykb.make_verbose()
    mykb.make_new_one_way_relation("Parent", "Alice", "Tom")
    mykb.make_new_one_way_relation("Parent", "Tom", "Charlie")
    mykb.make_new_rule("AND",[("ONE_WAY", "Parent", "x", "z"),("ONE_WAY","Parent", "z", "y")], ("ONE_WAY", "Ancestor", "x", "y")) #sub should have x = Alice, z = Tom, y = Charlie
    mykb.make_new_rule("AND",[("ONE_WAY", "Parent", "x", "y")], ("ONE_WAY", "Ancestor", "x", "y"))

    myfacts, myrules = mykb.get_mykb()

    print("Facts to be inputted: ", myfacts)
    print("Rules to be inputted: ", myrules)


    #tests

    #1: direct fact checking
    sub = {}
    print(mykb.derive_conclusion(("ONE_WAY", "Parent", "Alice", "Tom"), myfacts, myrules, sub)) #return true
    print(mykb.derive_conclusion(("ONE_WAY", "Parent", "Alice", "Zack"), myfacts, myrules, sub)) #return false

    #2: a deeper check involving rules
    sub = {}
    print(mykb.derive_conclusion(("ONE_WAY", "Ancestor", "Alice", "Charlie"), myfacts, myrules, sub)) #return true
    print(mykb.derive_conclusion(("ONE_WAY", "Ancestor", "Tom", "Charlie"), myfacts, myrules, sub)) #return true






            

        









    







