!(import! &self motto)
!(import! &self py_ops)
(= (user-query) "Hello world.")

!(assertEqual ((echo-agent) (Script basic_script.mps))
"user Hello world.
assistant Hi there")

!(remove-atom &self (= (user-query) "Hello world."))
(= (user-query) "Bye")
!(assertEqual ((echo-agent) (Script basic_script.mps))
"user Bye
assistant Bye")
